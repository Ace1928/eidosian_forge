from functools import reduce
import warnings
from Bio import BiopythonDeprecationWarning
def _calc_empirical_expects(xs, ys, classes, features):
    """Calculate the expectation of each function from the data (PRIVATE).

    This is the constraint for the maximum entropy distribution. Return a
    list of expectations, parallel to the list of features.
    """
    class2index = {}
    for index, key in enumerate(classes):
        class2index[key] = index
    ys_i = [class2index[y] for y in ys]
    expect = []
    N = len(xs)
    for feature in features:
        s = 0
        for i in range(N):
            s += feature.get((i, ys_i[i]), 0)
        expect.append(s / N)
    return expect