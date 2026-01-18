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
def _update_rs_from_primary(sds: MutableMapping[_Address, ServerDescription], replica_set_name: Optional[str], server_description: ServerDescription, max_set_version: Optional[int], max_election_id: Optional[ObjectId]) -> tuple[int, Optional[str], Optional[int], Optional[ObjectId]]:
    """Update topology description from a primary's hello response.

    Pass in a dict of ServerDescriptions, current replica set name, the
    ServerDescription we are processing, and the TopologyDescription's
    max_set_version and max_election_id if any.

    Returns (new topology type, new replica_set_name, new max_set_version,
    new max_election_id).
    """
    if replica_set_name is None:
        replica_set_name = server_description.replica_set_name
    elif replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
        return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)
    if server_description.max_wire_version is None or server_description.max_wire_version < 17:
        new_election_tuple: tuple = (server_description.set_version, server_description.election_id)
        max_election_tuple: tuple = (max_set_version, max_election_id)
        if None not in new_election_tuple:
            if None not in max_election_tuple and new_election_tuple < max_election_tuple:
                sds[server_description.address] = server_description.to_unknown()
                return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)
            max_election_id = server_description.election_id
        if server_description.set_version is not None and (max_set_version is None or server_description.set_version > max_set_version):
            max_set_version = server_description.set_version
    else:
        new_election_tuple = (server_description.election_id, server_description.set_version)
        max_election_tuple = (max_election_id, max_set_version)
        new_election_safe = tuple((MinKey() if i is None else i for i in new_election_tuple))
        max_election_safe = tuple((MinKey() if i is None else i for i in max_election_tuple))
        if new_election_safe < max_election_safe:
            sds[server_description.address] = server_description.to_unknown()
            return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)
        else:
            max_election_id = server_description.election_id
            max_set_version = server_description.set_version
    for server in sds.values():
        if server.server_type is SERVER_TYPE.RSPrimary and server.address != server_description.address:
            sds[server.address] = server.to_unknown()
            break
    for new_address in server_description.all_hosts:
        if new_address not in sds:
            sds[new_address] = ServerDescription(new_address)
    for addr in set(sds) - server_description.all_hosts:
        sds.pop(addr)
    return (_check_has_primary(sds), replica_set_name, max_set_version, max_election_id)