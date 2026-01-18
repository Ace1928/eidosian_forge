from __future__ import annotations
from random import sample
from typing import (
from bson.min_key import MinKey
from bson.objectid import ObjectId
from pymongo import common
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref, _ServerMode
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import Selection
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import _Address
def _apply_local_threshold(self, selection: Optional[Selection]) -> list[ServerDescription]:
    if not selection:
        return []
    fastest = min((cast(float, s.round_trip_time) for s in selection.server_descriptions))
    threshold = self._topology_settings.local_threshold_ms / 1000.0
    return [s for s in selection.server_descriptions if cast(float, s.round_trip_time) - fastest <= threshold]