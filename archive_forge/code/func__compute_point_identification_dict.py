from . import matrix
def _compute_point_identification_dict(choose_generators_info):
    """
    A vertex in the fundamental domain is an equivalence class of
    vertices (tet, v0, v1, v2) of doubly truncated simplicies under face
    gluings not corresponding to generators.

    This method computes the equivalence classes and returns them
    as dictionary mapping a vertex quadruple (tet, v0, v1, v2) to
    the set of equivalent triples.
    """
    d = dict([(Vertex(tet, v0, v1, v2), set([Vertex(tet, v0, v1, v2)])) for tet in range(len(choose_generators_info)) for v0, v1, v2, v3 in _perm4_iterator()])
    for this_tet, info in enumerate(choose_generators_info):
        for this_v0, this_v1, this_v2, this_v3 in _perm4_iterator():
            if info['generators'][this_v0] == 0:
                this_pt = Vertex(this_tet, this_v1, this_v2, this_v3)
                other_tet = info['neighbors'][this_v0]
                gluing = info['gluings'][this_v0]
                other_pt = Vertex(other_tet, gluing[this_v1], gluing[this_v2], gluing[this_v3])
                identified_pts = d[this_pt] | d[other_pt]
                for pt in identified_pts:
                    d[pt] = identified_pts
    return d