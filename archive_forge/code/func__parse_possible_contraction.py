import itertools
def _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost):
    """Copied from _parse_possible_contraction in numpy/core/einsumfunc.py

    Compute the cost (removed size + flops) and resultant indices for
    performing the contraction specified by ``positions``.

    Parameters
    ----------
    positions : tuple of int
        The locations of the proposed tensors to contract.
    input_sets : list of sets
        The indices found on each tensors.
    output_set : set
        The output indices of the expression.
    idx_dict : dict
        Mapping of each index to its size.
    memory_limit : int
        The total allowed size for an intermediary tensor.
    path_cost : int
        The contraction cost so far.
    naive_cost : int
        The cost of the unoptimized expression.

    Returns
    -------
    cost : (int, int)
        A tuple containing the size of any indices removed, and the flop cost.
    positions : tuple of int
        The locations of the proposed tensors to contract.
    new_input_sets : list of sets
        The resulting new list of indices if this proposed contraction is performed.

    """
    contract = _find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    new_size = _compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None
    old_sizes = (_compute_size_by_dict(input_sets[p], idx_dict) for p in positions)
    removed_size = sum(old_sizes) - new_size
    cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)
    if path_cost + cost > naive_cost:
        return None
    return [sort, positions, new_input_sets]