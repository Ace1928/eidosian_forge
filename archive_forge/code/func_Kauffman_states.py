def Kauffman_states(K):
    """Returns the set of Kauffman states for the Alexander polynomial, corresponding to the spanning trees in the black graph. Returns a list of dictionaries, with keys crossings and values faces in the knot projection."""
    G = K.black_graph()
    trees = G.spanning_trees()
    marked_edge = ((K.crossings[0], 0), K.crossings[0].adjacent[0])
    states = list()
    for T in trees:
        for v in T.vertices():
            if marked_edge[0] in v:
                root = v
        oT = orient_tree(T, root)
        dT = _dual_spanning_tree(K, T)
        for v in dT.vertices():
            if marked_edge[0] in v:
                droot = v
        odT = orient_tree(dT, droot)
        state = dict()
        for e in oT.edges():
            state[e[2]] = e[1]
        for e in odT.edges():
            state[e[2]] = e[1]
        states.append(state)
    return states