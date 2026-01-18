from __future__ import annotations
from collections import defaultdict
import copy
import glob
import os
import re
import time
from typing import Any, Iterable, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import strutils
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator import utils as initiator_utils
from os_brick import utils
def _get_ips_iqns_luns(self, connection_properties: dict, discover: bool=True, is_disconnect_call: bool=False) -> list[tuple[Any, Any, Any]]:
    """Build a list of ips, iqns, and luns.

        Used when doing singlepath and multipath, and we have 4 cases:

        - All information is in the connection properties
        - We have to do an iSCSI discovery to get the information
        - We don't want to do another discovery and we query the discoverydb
        - Discovery failed because it was actually a single pathed attachment

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
        :type connection_properties: dict
        :param discover: Whether doing an iSCSI discovery is acceptable.
        :type discover: bool
        :param is_disconnect_call: Whether this is a call coming from a user
                                   disconnect_volume call or a call from some
                                   other operation's cleanup.
        :type is_disconnect_call: bool
        :returns: list of tuples of (ip, iqn, lun)
        """
    try:
        if 'target_portals' in connection_properties and 'target_iqns' in connection_properties:
            ips_iqns_luns = self._get_all_targets(connection_properties)
        else:
            method = self._discover_iscsi_portals if discover else self._get_discoverydb_portals
            ips_iqns_luns = method(connection_properties)
    except exception.TargetPortalNotFound:
        if is_disconnect_call:
            return self._get_all_targets(connection_properties)
        raise
    except Exception:
        LOG.exception('Exception encountered during portal discovery')
        if 'target_portals' in connection_properties:
            raise exception.TargetPortalsNotFound(target_portals=connection_properties['target_portals'])
        if 'target_portal' in connection_properties:
            raise exception.TargetPortalNotFound(target_portal=connection_properties['target_portal'])
        raise
    if not connection_properties.get('target_iqns'):
        main_iqn = connection_properties['target_iqn']
        all_portals = {(ip, lun) for ip, iqn, lun in ips_iqns_luns}
        match_portals = {(ip, lun) for ip, iqn, lun in ips_iqns_luns if iqn == main_iqn}
        if len(all_portals) == len(match_portals):
            ips_iqns_luns = [(p[0], main_iqn, p[1]) for p in all_portals]
    return ips_iqns_luns