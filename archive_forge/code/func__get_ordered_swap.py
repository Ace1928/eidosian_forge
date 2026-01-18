def _get_ordered_swap(permutation_in):
    """Sorts the input permutation by iterating through the permutation list
    and putting each element to its correct position via a SWAP (if it's not
    at the correct position already). If ``n`` is the length of the input
    permutation, this requires at most ``n`` SWAPs.

    More precisely, if the input permutation is a cycle of length ``m``,
    then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
    if the input  permutation consists of several disjoint cycles, then each cycle
    is essentially treated independently.
    """
    permutation = list(permutation_in[:])
    swap_list = []
    index_map = _inverse_pattern(permutation_in)
    for i, val in enumerate(permutation):
        if val != i:
            j = index_map[i]
            swap_list.append((i, j))
            permutation[i], permutation[j] = (permutation[j], permutation[i])
            index_map[val] = j
            index_map[i] = i
    swap_list.reverse()
    return swap_list