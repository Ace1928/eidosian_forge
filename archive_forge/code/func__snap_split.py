from collections import Counter, defaultdict
import networkx as nx
def _snap_split(groups, neighbor_info, group_lookup, group_id):
    """
    Splits a group based on edge types and updates the groups accordingly

    Splits the group with the given group_id based on the edge types
    of the nodes so that each new grouping will all have the same
    edges with other nodes.

    Parameters
    ----------
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    neighbor_info: dict
        A data structure indicating the number of edges a node has with the
        groups in the current summarization of each edge type
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization
    group_lookup: dict
        dictionary of nodes and their current corresponding group ID
    group_id: object
        ID of group to be split

    Returns
    -------
    dict
        The updated groups based on the split
    """
    new_group_mappings = defaultdict(set)
    for node in groups[group_id]:
        signature = tuple((frozenset(edge_types) for edge_types in neighbor_info[node].values()))
        new_group_mappings[signature].add(node)
    new_groups = sorted(new_group_mappings.values(), key=len)
    for new_group in new_groups[:-1]:
        new_group_id = len(groups)
        groups[new_group_id] = new_group
        groups[group_id] -= new_group
        for node in new_group:
            group_lookup[node] = new_group_id
    return groups