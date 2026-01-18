from __future__ import annotations
import errno
import functools
import glob
import json
import os.path
import time
from typing import (Callable, Optional, Sequence, Type, Union)  # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
class NVMeOFConnProps(object):
    """Internal representation of the NVMe-oF connection properties

    There is an old and a newer connection properties format, which result
    in 2 variants for replicated connections and 2 for non replicated:

    1- New format with multiples replicas information
    2- New format with single replica information
    3- New format with no replica information
    4- Original format

    Case #1 and #2 format:
      {
       'vol_uuid': <cinder_volume_id>,
       'alias': <raid_alias>,
       'volume_replicas': [ <target>, ... ],
       'replica_count': len(volume_replicas),
      }

      Where:
        cinder_volume_id ==> Cinder id, could be different from NVMe UUID.
                             with/without hyphens, uppper/lower cased.
        target :== {
                    'target_nqn': <target_nqn>,
                    'vol_uuid': <nvme_volume_uuid>,
                    'portals': [ <portal>, ... ],
                   }

        nvme_volume_uuid ==>  NVMe UUID. Can be different than cinder's id.
                              With/without hyphens, uppper/lower cased
        portal ::= tuple/list(
                    <target_portal>,
                    <target_port>,
                    <transport_type>
                   )
        transport_type ::= 'RoCEv2' | <anything>  # anything => tcp

    Case #3 format:
      <target>  ==> As defined in case #1 & #2

    Case #4 format:
      {
       'nqn': <nqn>,
       'transport_type': <transport_type>,
       'target_portal': <target_address>,
       'target_port': <target_port>,
       'volume_nguid': <volume_nguid>,
       'ns_id': <target_namespace_id>,
       'host_nqn': <connector_host_nqn>,
      }

      Where:
        transport_type ::= 'rdma' | 'tcp'
        volume_nguid ==> Optional, with/without hyphens, uppper/lower cased
        target_namespace_id ==> Optional
        connector_host_nqn> ==> Optional

    This class unifies the representation of all these in the following
    attributes:

      replica_count: None for non replicated
      alias: None for non replicated
      cinder_volume_id: None for non replicated
      is_replicated: True if replica count > 1, None if count = 1 else False
      targets: List of Target instances
      device_path: None if not present (it's set by Nova)

    In this unification case#4 is treated as case#3 where the vol_uuid is None
    and leaving all the additional information in the dictionary.  This way non
    replicated cases are always handled in the same way and we have a common
    <target>" definition for all cases:

      target :== {
                  'target_nqn': <target_nqn>,
                  'vol_uuid': <nvme_volume_uuid>,
                  'portals': [ <new_portal>, ... ],
                  'volume_nguid': <volume_nguid>,
                  'ns_id': <target_namespace_id>,
                  'host_nqn': <connector_host_nqn>,
                 }
      new_portal ::= tuple/list(
                      <target_address>,
                      <target_port>,
                      <new_transport_type>
                     )
      new_transport_type ::= 'rdma' | 'tcp'

    Portals change the transport_type to the internal representation where:
        'RoCEv2' ==> 'rdma'
        <else> ==> 'tcp'

    This means that the new connection format now accepts vol_uuid set to None,
    and accepts ns_id, volume_nguid, and host_nqn parameters, as described in
    the connect_volume docstring.
    """
    RO = 'ro'
    RW = 'rw'
    replica_count = None
    cinder_volume_id: Optional[str] = None

    def __init__(self, conn_props: dict, find_controllers: bool=False) -> None:
        self.qos_specs = conn_props.get('qos_specs')
        self.readonly = conn_props.get('access_mode', self.RW) == self.RO
        self.encrypted = conn_props.get('encrypted', False)
        self.cacheable = conn_props.get('cacheable', False)
        self.discard = conn_props.get('discard', False)
        if REPLICAS not in conn_props and NQN not in conn_props:
            LOG.debug('Converting old connection info to new format')
            conn_props[UUID] = None
            conn_props[NQN] = conn_props.pop(OLD_NQN)
            conn_props[PORTALS] = [(conn_props.pop(PORTAL), conn_props.pop(PORT), conn_props.pop(TRANSPORT))]
        self.alias = conn_props.get(ALIAS)
        if REPLICAS in conn_props:
            self.replica_count = conn_props[REPLICA_COUNT] or len(conn_props[REPLICAS])
            self.is_replicated = True if self.replica_count > 1 else None
            targets = conn_props[REPLICAS]
            self.cinder_volume_id = str(uuid_lib.UUID(conn_props[UUID]))
        else:
            self.is_replicated = False
            targets = [conn_props]
        self.targets = [Target.factory(source_conn_props=self, find_controllers=find_controllers, **target) for target in targets]
        self.device_path = conn_props.get('device_path')

    def get_devices(self, only_live: bool=False) -> list[str]:
        """Get all device paths present in the system for all targets."""
        result = []
        for target in self.targets:
            result.extend(target.get_devices(only_live))
        return result

    @classmethod
    def from_dictionary_parameter(cls: Type['NVMeOFConnProps'], func: Callable) -> Callable:
        """Decorator to convert connection properties dictionary.

        It converts the connection properties into a NVMeOFConnProps instance
        and finds the controller names for all portals present in the system.
        """

        @functools.wraps(func)
        def wrapper(self, connection_properties, *args, **kwargs):
            conn_props = cls(connection_properties, find_controllers=True)
            return func(self, conn_props, *args, **kwargs)
        return wrapper