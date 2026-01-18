from __future__ import annotations
from typing import TYPE_CHECKING
from pymongo.errors import ConfigurationError
from pymongo.server_type import SERVER_TYPE
def _no_primary(max_staleness: int, selection: Selection) -> Selection:
    """Apply max_staleness, in seconds, to a Selection with no known primary."""
    smax = selection.secondary_with_max_last_write_date()
    if not smax:
        return selection.with_server_descriptions([])
    sds = []
    for s in selection.server_descriptions:
        if s.server_type == SERVER_TYPE.RSSecondary:
            assert smax.last_write_date and s.last_write_date
            staleness = smax.last_write_date - s.last_write_date + selection.heartbeat_frequency
            if staleness <= max_staleness:
                sds.append(s)
        else:
            sds.append(s)
    return selection.with_server_descriptions(sds)