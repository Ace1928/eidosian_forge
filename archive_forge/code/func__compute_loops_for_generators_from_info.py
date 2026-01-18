from . import matrix
def _compute_loops_for_generators_from_info(choose_generators_info, point_to_shortest_path, penalties):
    """
    Using the result of SnapPy's _choose_generators_info() that
    indicates which face pairings correspond to which generators and
    the shortest path dictionary computed previously,
    compute a loop in the short and long edges for each generator.
    """
    num_generators = _compute_num_generators(choose_generators_info)
    loops_for_generators = num_generators * [None]
    for this_tet, info in enumerate(choose_generators_info):
        for this_v0, this_v1, this_v2, this_v3 in _perm4_iterator():
            generator_index = info['generators'][this_v0]
            if generator_index > 0:
                this_pt = Vertex(this_tet, this_v1, this_v2, this_v3)
                other_tet = info['neighbors'][this_v0]
                gluing = info['gluings'][this_v0]
                other_pt = Vertex(other_tet, gluing[this_v1], gluing[this_v2], gluing[this_v3])
                new_loop = point_to_shortest_path[this_pt] * point_to_shortest_path[other_pt] ** (-1)
                loop = loops_for_generators[generator_index - 1]
                if loop is None or _penalty_of_path(new_loop, penalties) < _penalty_of_path(loop, penalties):
                    loops_for_generators[generator_index - 1] = new_loop
    return loops_for_generators