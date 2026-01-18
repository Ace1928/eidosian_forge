import numpy as np
def S_within_hac(x, d, nlags=1, groupidx=1):
    """HAC for observations within a categorical group
    """
    r = np.zeros(d.shape[1])
    r[0] = 1
    r[groupidx] = 1
    weights = weights_bartlett(nlags)
    return aggregate_cov(x, d, r=r, weights=weights)