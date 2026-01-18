import numpy as np
import scipy.stats
import cirq
def _random_unitary():
    return scipy.stats.unitary_group.rvs(2)