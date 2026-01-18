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
def _update_rs_with_primary_from_member(sds: MutableMapping[_Address, ServerDescription], replica_set_name: Optional[str], server_description: ServerDescription) -> int:
    """RS with known primary. Process a response from a non-primary.

    Pass in a dict of ServerDescriptions, current replica set name, and the
    ServerDescription we are processing.

    Returns new topology type.
    """
    assert replica_set_name is not None
    if replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
    elif server_description.me and server_description.address != server_description.me:
        sds.pop(server_description.address)
    return _check_has_primary(sds)