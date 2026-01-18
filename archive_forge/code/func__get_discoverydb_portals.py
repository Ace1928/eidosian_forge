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
def _get_discoverydb_portals(self, connection_properties: dict) -> list[tuple]:
    """Retrieve iscsi portals information from the discoverydb.

        Example of discoverydb command output:

        SENDTARGETS:
        DiscoveryAddress: 192.168.1.33,3260
        DiscoveryAddress: 192.168.1.2,3260
        Target: iqn.2004-04.com.qnap:ts-831x:iscsi.cinder-20170531114245.9eff88
            Portal: 192.168.1.3:3260,1
                Iface Name: default
            Portal: 192.168.1.2:3260,1
                Iface Name: default
        Target: iqn.2004-04.com.qnap:ts-831x:iscsi.cinder-20170531114447.9eff88
            Portal: 192.168.1.3:3260,1
                Iface Name: default
            Portal: 192.168.1.2:3260,1
                Iface Name: default
        DiscoveryAddress: 192.168.1.38,3260
        iSNS:
        No targets found.
        STATIC:
        No targets found.
        FIRMWARE:
        No targets found.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
        :type connection_properties: dict
        :returns: list of tuples of (ip, iqn, lun)
        """
    ip, port = connection_properties['target_portal'].rsplit(':', 1)
    ip = ip.replace('[', '\\[?').replace(']', '\\]?')
    out = self._run_iscsiadm_bare(['-m', 'discoverydb', '-o', 'show', '-P', 1])[0] or ''
    regex = ''.join(('^SENDTARGETS:\n.*?^DiscoveryAddress: ', ip, ',', port, '.*?\n(.*?)^(?:DiscoveryAddress|iSNS):.*'))
    LOG.debug('Regex to get portals from discoverydb: %s', regex)
    info = re.search(regex, out, re.DOTALL | re.MULTILINE)
    ips = []
    iqns = []
    if info:
        iscsi_transport = 'iser' if self._get_transport() == 'iser' else 'default'
        iface = 'Iface Name: ' + iscsi_transport
        current_iqn = ''
        current_ip = ''
        for line in info.group(1).splitlines():
            line = line.strip()
            if line.startswith('Target:'):
                current_iqn = line[8:]
            elif line.startswith('Portal:'):
                current_ip = line[8:].split(',')[0]
            elif line.startswith(iface):
                if current_iqn and current_ip:
                    iqns.append(current_iqn)
                    ips.append(current_ip)
                current_ip = ''
    if not iqns:
        raise exception.TargetPortalsNotFound(_('Unable to find target portals information on discoverydb.'))
    luns = self._get_luns(connection_properties, iqns)
    return list(zip(ips, iqns, luns))