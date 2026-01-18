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
def _check_has_primary(sds: Mapping[_Address, ServerDescription]) -> int:
    """Current topology type is ReplicaSetWithPrimary. Is primary still known?

    Pass in a dict of ServerDescriptions.

    Returns new topology type.
    """
    for s in sds.values():
        if s.server_type == SERVER_TYPE.RSPrimary:
            return TOPOLOGY_TYPE.ReplicaSetWithPrimary
    else:
        return TOPOLOGY_TYPE.ReplicaSetNoPrimary