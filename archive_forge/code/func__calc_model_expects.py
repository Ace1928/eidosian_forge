from functools import reduce
import warnings
from Bio import BiopythonDeprecationWarning
def _calc_model_expects(xs, classes, features, alphas):
    """Calculate the expectation of each feature from the model (PRIVATE).

    This is not used in maximum entropy training, but provides a good function
    for debugging.
    """
    p_yx = _calc_p_class_given_x(xs, classes, features, alphas)
    expects = []
    for feature in features:
        sum = 0.0
        for (i, j), f in feature.items():
            sum += p_yx[i][j] * f
        expects.append(sum / len(xs))
    return expects