import math
def _preprocess_for_efficient_roulette_selection(discretized_probabilities):
    """Prepares data for performing efficient roulette selection.

    The output is a tuple (alternates, keep_weights). The output is guaranteed
    to satisfy a sampling-equivalence property. Specifically, the following
    sampling process is guaranteed to be equivalent to simply picking index i
    with probability weights[i] / sum(weights):

        1. Pick a number i in [0, len(weights) - 1] uniformly at random.
        2. Return i With probability keep_weights[i]*len(weights)/sum(weights).
        3. Otherwise return alternates[i].

    In other words, the output makes it possible to perform roulette selection
    while generating only two random numbers, doing a single lookup of the
    relevant (keep_chance, alternate) pair, and doing one comparison. This is
    not so useful classically, but in the context of a quantum computation
    where all those things are expensive the second sampling process is far
    superior.

    Args:
        discretized_probabilities: A list of probabilities approximated by
            integer numerators (with an implied common denominator). In order
            to operate without floating point error, it is required that the
            sum of this list is a multiple of the number of items in the list.

    Returns:
        alternates (list[int]): An alternate index for each index from 0 to
            len(weights) - 1
        keep_weight (list[int]): Indicates how often one should stay at index i
            instead of switching to alternates[i]. To get the actual keep
            probability of the i'th element, multiply keep_weight[i] by
            len(discretized_probabilities) then divide by
            sum(discretized_probabilities).
    Raises:
        ValueError: if `discretized_probabilities` input is empty or if the sum of elements
            in the list is not a multiple of the number of items in the list.
    """
    weights = list(discretized_probabilities)
    if not weights:
        raise ValueError('Empty input.')
    n = len(weights)
    target_weight = sum(weights) // n
    if sum(weights) != n * target_weight:
        raise ValueError('sum(weights) must be a multiple of len(weights).')
    alternates = list(range(n))
    keep_weights = [0] * n
    donor_position = 0
    for _ in range(2):
        for i in range(n):
            if weights[i] >= target_weight:
                continue
            while weights[donor_position] <= target_weight:
                donor_position += 1
            donated = target_weight - weights[i]
            weights[donor_position] -= donated
            alternates[i] = donor_position
            keep_weights[i] = weights[i]
            weights[i] = target_weight
    return (alternates, keep_weights)