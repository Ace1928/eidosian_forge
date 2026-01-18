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
def _get_connection_devices(self, connection_properties: dict, ips_iqns_luns: Optional[list[tuple[str, str, str]]]=None, is_disconnect_call: bool=False) -> dict[set, set]:
    """Get map of devices by sessions from our connection.

        For each of the TCP sessions that correspond to our connection
        properties we generate a map of (ip, iqn) to (belong, other) where
        belong is a set of devices in that session that populated our system
        when we did a connection using connection properties, and other are
        any other devices that share that same session but are the result of
        connecting with different connection properties.

        We also include all nodes from our connection that don't have a
        session.

        If ips_iqns_luns parameter is provided connection_properties won't be
        used to get them.

        When doing multipath we may not have all the information on the
        connection properties (sendtargets was used on connect) so we may have
        to retrieve the info from the discoverydb.  Call _get_ips_iqns_luns to
        do the right things.

        This method currently assumes that it's only called by the
        _cleanup_conection method.
        """
    if not ips_iqns_luns:
        ips_iqns_luns = self._get_ips_iqns_luns(connection_properties, discover=False, is_disconnect_call=is_disconnect_call)
    LOG.debug('Getting connected devices for (ips,iqns,luns)=%s', ips_iqns_luns)
    nodes = self._get_iscsi_nodes()
    sessions = self._get_iscsi_sessions_full()
    sessions_map = {(s[2], s[4]): s[1] for s in sessions if s[0] in self.VALID_SESSIONS_PREFIX}
    device_map: defaultdict = defaultdict(lambda: (set(), set()))
    for ip, iqn, lun in ips_iqns_luns:
        session = sessions_map.get((ip, iqn))
        if not session:
            if (ip, iqn) in nodes:
                device_map[ip, iqn] = (set(), set())
            continue
        paths = glob.glob('/sys/class/scsi_host/host*/device/session' + session + '/target*/*:*:*:*/block/*')
        belong, others = device_map[ip, iqn]
        for path in paths:
            __, hctl, __, device = path.rsplit('/', 3)
            lun_path = int(hctl.rsplit(':', 1)[-1])
            device = device.strip('0123456789')
            if lun_path == lun:
                belong.add(device)
            else:
                others.add(device)
    LOG.debug('Resulting device map %s', device_map)
    return device_map