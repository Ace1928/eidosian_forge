import numpy as np
def S_all_hac(x, d, nlags=1):
    """HAC independent of categorical group membership
    """
    r = np.zeros(d.shape[1])
    r[0] = 1
    weights = weights_bartlett(nlags)
    return aggregate_cov(x, d, r=r, weights=weights)