import numpy as np
def S_cluster(x, d, groupidx=[1]):
    r = np.zeros(d.shape[1])
    r[groupidx] = 1
    return aggregate_cov(x, d, r=r, weights=None)