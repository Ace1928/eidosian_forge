from collections import Counter, defaultdict
import networkx as nx
def _snap_eligible_group(G, groups, group_lookup, edge_types):
    """
    Determines if a group is eligible to be split.

    A group is eligible to be split if all nodes in the group have edges of the same type(s)
    with the same other groups.

    Parameters
    ----------
    G: graph
        graph to be summarized
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    group_lookup: dict
        dictionary of nodes and their current corresponding group ID
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization

    Returns
    -------
    tuple: group ID to split, and neighbor-groups participation_counts data structure
    """
    neighbor_info = {node: {gid: Counter() for gid in groups} for node in group_lookup}
    for group_id in groups:
        current_group = groups[group_id]
        for node in current_group:
            neighbor_info[node] = {group_id: Counter() for group_id in groups}
            edges = G.edges(node, keys=True) if G.is_multigraph() else G.edges(node)
            for edge in edges:
                neighbor = edge[1]
                edge_type = edge_types[edge]
                neighbor_group_id = group_lookup[neighbor]
                neighbor_info[node][neighbor_group_id][edge_type] += 1
        group_size = len(current_group)
        for other_group_id in groups:
            edge_counts = Counter()
            for node in current_group:
                edge_counts.update(neighbor_info[node][other_group_id].keys())
            if not all((count == group_size for count in edge_counts.values())):
                return (group_id, neighbor_info)
    return (None, neighbor_info)