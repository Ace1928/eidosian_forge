import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
def _survdiff(time, status, group, weight_type, gr, entry=None, **kwargs):
    if entry is None:
        utimes, rtimes = np.unique(time, return_inverse=True)
    else:
        utimes, rtimes = np.unique(np.concatenate((time, entry)), return_inverse=True)
        rtimes = rtimes[0:len(time)]
    tse = [(gr_i, None) for gr_i in gr]
    if entry is not None:
        for k, _ in enumerate(gr):
            ii = group == gr[k]
            entry1 = entry[ii]
            tse[k] = (gr[k], entry1)
    nrisk, obsv = ([], [])
    ml = len(utimes)
    for g, entry0 in tse:
        mk = group == g
        n = np.bincount(rtimes, weights=mk, minlength=ml)
        ob = np.bincount(rtimes, weights=status * mk, minlength=ml)
        obsv.append(ob)
        if entry is not None:
            n = np.cumsum(n) - n
            rentry = np.searchsorted(utimes, entry0, side='left')
            n0 = np.bincount(rentry, minlength=ml)
            n0 = np.cumsum(n0) - n0
            nr = n0 - n
        else:
            nr = np.cumsum(n[::-1])[::-1]
        nrisk.append(nr)
    obs = sum(obsv)
    nrisk_tot = sum(nrisk)
    ix = np.flatnonzero(nrisk_tot > 1)
    weights = None
    if weight_type is not None:
        weight_type = weight_type.lower()
        if weight_type == 'gb':
            weights = nrisk_tot
        elif weight_type == 'tw':
            weights = np.sqrt(nrisk_tot)
        elif weight_type == 'fh':
            if 'fh_p' not in kwargs:
                msg = "weight_type type 'fh' requires specification of fh_p"
                raise ValueError(msg)
            fh_p = kwargs['fh_p']
            sp = 1 - obs / nrisk_tot.astype(np.float64)
            sp = np.log(sp)
            sp = np.cumsum(sp)
            sp = np.exp(sp)
            weights = sp ** fh_p
            weights = np.roll(weights, 1)
            weights[0] = 1
        else:
            raise ValueError('weight_type not implemented')
    dfs = len(gr) - 1
    r = np.vstack(nrisk) / np.clip(nrisk_tot, 1e-10, np.inf)[None, :]
    groups_oe = []
    groups_var = []
    var_denom = nrisk_tot - 1
    var_denom = np.clip(var_denom, 1e-10, np.inf)
    for g in range(1, dfs + 1):
        oe = obsv[g] - r[g] * obs
        var_tensor_part = r[1:, :].T * (np.eye(1, dfs, g - 1).ravel() - r[g, :, None])
        var_scalar_part = obs * (nrisk_tot - obs) / var_denom
        var = var_tensor_part * var_scalar_part[:, None]
        if weights is not None:
            oe = weights * oe
            var = (weights ** 2)[:, None] * var
        groups_oe.append(oe[ix].sum())
        groups_var.append(var[ix].sum(axis=0))
    obs_vec = np.hstack(groups_oe)
    var_mat = np.vstack(groups_var)
    return (obs_vec, var_mat)