from itertools import groupby
from warnings import warn
import numpy as np
from scipy.sparse import find, coo_matrix
def _sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups):
    n = y.shape[0]
    n_groups = np.max(groups) + 1
    h_vecs = np.empty((n_groups, n))
    for group in range(n_groups):
        e = np.equal(group, groups)
        h_vecs[group] = h * e
    h_vecs = h_vecs.T
    f_new = fun(t, y[:, None] + h_vecs)
    df = f_new - f[:, None]
    i, j, _ = find(structure)
    diff = coo_matrix((df[i, groups[j]], (i, j)), shape=(n, n)).tocsc()
    max_ind = np.array(abs(diff).argmax(axis=0)).ravel()
    r = np.arange(n)
    max_diff = np.asarray(np.abs(diff[max_ind, r])).ravel()
    scale = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, groups[r]]))
    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        ind, = np.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = y[ind] + new_factor * y_scale[ind] - y[ind]
        h_new_all = np.zeros(n)
        h_new_all[ind] = h_new
        groups_unique = np.unique(groups[ind])
        groups_map = np.empty(n_groups, dtype=int)
        h_vecs = np.empty((groups_unique.shape[0], n))
        for k, group in enumerate(groups_unique):
            e = np.equal(group, groups)
            h_vecs[k] = h_new_all * e
            groups_map[group] = k
        h_vecs = h_vecs.T
        f_new = fun(t, y[:, None] + h_vecs)
        df = f_new - f[:, None]
        i, j, _ = find(structure[:, ind])
        diff_new = coo_matrix((df[i, groups_map[groups[ind[j]]]], (i, j)), shape=(n, ind.shape[0])).tocsc()
        max_ind_new = np.array(abs(diff_new).argmax(axis=0)).ravel()
        r = np.arange(ind.shape[0])
        max_diff_new = np.asarray(np.abs(diff_new[max_ind_new, r])).ravel()
        scale_new = np.maximum(np.abs(f[max_ind_new]), np.abs(f_new[max_ind_new, groups_map[groups[ind]]]))
        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            update, = np.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]
    diff.data /= np.repeat(h, np.diff(diff.indptr))
    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)
    return (diff, factor)