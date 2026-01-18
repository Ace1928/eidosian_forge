from . import matrix
def _compute_point_to_shortest_path(point_identification_dict, origin, penalties):
    """
    Given the equivalence classes of quadruples (tet, v0, v1, v2) representing
    the same vertex in the fundamental domain and an origin,
    compute a shortest path from the origin to each vertex.

    This is returned as a dictionary mapping each triple to a shortest path.
    Triples corresponding to the same identified vertex all have the same
    path.
    """
    d = {}

    def identified_points_to_path(pt, path):
        return dict([(identified_pt, path) for identified_pt in point_identification_dict[pt]])
    previously_added = identified_points_to_path(origin, Path())
    while previously_added:
        d.update(previously_added)
        new_paths = {}
        for pt, path in sorted(previously_added.items()):
            for edge in pt.edges_starting_at_vertex():
                new_path = path * edge
                new_end_point = edge.end_point()
                if new_end_point not in d or _penalty_of_path(new_path, penalties) < _penalty_of_path(d[new_end_point], penalties):
                    new_paths.update(identified_points_to_path(new_end_point, new_path))
        previously_added = new_paths
    return d