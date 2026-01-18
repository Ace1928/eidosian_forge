from __future__ import annotations
import os
import typing
from typing import Any, Optional  # noqa: H301
from oslo_log import log as logging
from oslo_service import loopingcall
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator import linuxfc
from os_brick import utils
def _add_targets_to_connection_properties(self, connection_properties: dict) -> dict:
    LOG.debug('Adding targets to connection properties receives: %s', connection_properties)
    target_wwn = connection_properties.get('target_wwn')
    target_wwns = connection_properties.get('target_wwns')
    if target_wwns:
        wwns = target_wwns
    elif isinstance(target_wwn, list):
        wwns = target_wwn
    elif isinstance(target_wwn, str):
        wwns = [target_wwn]
    else:
        wwns = []
    wwns = [wwn.lower() for wwn in wwns]
    if target_wwns:
        connection_properties['target_wwns'] = wwns
    elif target_wwn:
        connection_properties['target_wwn'] = wwns
    target_lun = connection_properties.get('target_lun', 0)
    target_luns = connection_properties.get('target_luns')
    if target_luns:
        luns = target_luns
    elif isinstance(target_lun, int):
        luns = [target_lun]
    else:
        luns = []
    if len(luns) == len(wwns):
        targets = list(zip(wwns, luns))
    elif len(luns) == 1 and len(wwns) > 1:
        targets = [(wwn, luns[0]) for wwn in wwns]
    else:
        msg = _('Unable to find potential volume paths for FC device with luns: %(luns)s and wwns: %(wwns)s.') % {'luns': luns, 'wwns': wwns}
        LOG.error(msg)
        raise exception.VolumePathsNotFound(msg)
    connection_properties['targets'] = targets
    wwpn_lun_map = {wwpn: lun for wwpn, lun in targets}
    if connection_properties.get('initiator_target_map') is not None:
        itmap = connection_properties['initiator_target_map']
        itmap = {k.lower(): [port.lower() for port in v] for k, v in itmap.items()}
        connection_properties['initiator_target_map'] = itmap
        itmaplun = dict()
        for init_wwpn, target_wwpns in itmap.items():
            itmaplun[init_wwpn] = [(target_wwpn, wwpn_lun_map[target_wwpn]) for target_wwpn in target_wwpns if target_wwpn in wwpn_lun_map]
            if len(itmaplun[init_wwpn]) != len(itmap[init_wwpn]):
                unknown = set(itmap[init_wwpn])
                unknown.difference_update(itmaplun[init_wwpn])
                LOG.warning('Driver returned an unknown targets in the initiator mapping %s', ', '.join(unknown))
        connection_properties['initiator_target_lun_map'] = itmaplun
    LOG.debug('Adding targets to connection properties returns: %s', connection_properties)
    return connection_properties