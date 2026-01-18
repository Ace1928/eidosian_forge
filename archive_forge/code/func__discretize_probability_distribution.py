import math
def _discretize_probability_distribution(unnormalized_probabilities, epsilon):
    """Approximates probabilities with integers over a common denominator.

    Args:
        unnormalized_probabilities: A list of non-negative floats proportional
            to probabilities from a probability distribution. The numbers may
            not be normalized (they do not have to add up to 1).
        epsilon: The absolute error tolerance.

    Returns:
        numerators (list[int]): A list of numerators for each probability.
        denominator (int): The common denominator to divide numerators by to
            get probabilities.
        sub_bit_precision (int): The exponent mu such that
            denominator = n * 2**mu
            where n = len(unnormalized_probabilities).

        It is guaranteed that numerators[i] / denominator is within epsilon of
        the i'th input probability (after normalization).
    """
    n = len(unnormalized_probabilities)
    sub_bit_precision = max(0, int(math.ceil(-math.log(epsilon * n, 2))))
    bin_count = 2 ** sub_bit_precision * n
    cumulative = list(_partial_sums(unnormalized_probabilities))
    total = cumulative[-1]
    discretized_cumulative = [int(math.floor(c / total * bin_count + 0.5)) for c in cumulative]
    discretized = list(_differences(discretized_cumulative))
    return (discretized, bin_count, sub_bit_precision)