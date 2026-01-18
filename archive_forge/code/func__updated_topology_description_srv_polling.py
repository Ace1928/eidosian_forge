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
def _updated_topology_description_srv_polling(topology_description: TopologyDescription, seedlist: list[tuple[str, Any]]) -> TopologyDescription:
    """Return an updated copy of a TopologyDescription.

    :Parameters:
      - `topology_description`: the current TopologyDescription
      - `seedlist`: a list of new seeds new ServerDescription that resulted from
        a hello call
    """
    assert topology_description.topology_type in SRV_POLLING_TOPOLOGIES
    sds = topology_description.server_descriptions()
    if set(sds.keys()) == set(seedlist):
        return topology_description
    for address in list(sds.keys()):
        if address not in seedlist:
            sds.pop(address)
    if topology_description.srv_max_hosts != 0:
        new_hosts = set(seedlist) - set(sds.keys())
        n_to_add = topology_description.srv_max_hosts - len(sds)
        if n_to_add > 0:
            seedlist = sample(sorted(new_hosts), min(n_to_add, len(new_hosts)))
        else:
            seedlist = []
    for address in seedlist:
        if address not in sds:
            sds[address] = ServerDescription(address)
    return TopologyDescription(topology_description.topology_type, sds, topology_description.replica_set_name, topology_description.max_set_version, topology_description.max_election_id, topology_description._topology_settings)