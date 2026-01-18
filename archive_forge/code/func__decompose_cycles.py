def _decompose_cycles(cycles):
    """Given a disjoint cycle decomposition, decomposes every cycle into a SWAP
    circuit of depth 2."""
    swap_list = []
    for cycle in cycles:
        m = len(cycle)
        for i in range((m - 1) // 2):
            swap_list.append((cycle[i - 1], cycle[m - 3 - i]))
        for i in range(m // 2):
            swap_list.append((cycle[i - 1], cycle[m - 2 - i]))
    return swap_list