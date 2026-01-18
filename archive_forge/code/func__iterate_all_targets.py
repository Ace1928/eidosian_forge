from __future__ import annotations
import copy
from typing import Any, Generator  # noqa: H301
from os_brick.initiator import initiator_connector
def _iterate_all_targets(self, connection_properties: dict) -> Generator[dict[str, Any], None, None]:
    for portal, iqn, lun in self._get_all_targets(connection_properties):
        props = copy.deepcopy(connection_properties)
        props['target_portal'] = portal
        props['target_iqn'] = iqn
        props['target_lun'] = lun
        for key in ('target_portals', 'target_iqns', 'target_luns'):
            props.pop(key, None)
        yield props