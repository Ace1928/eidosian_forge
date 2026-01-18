import builtins
from warnings import catch_warnings, simplefilter
import numpy as np
from operator import index
from collections import namedtuple
def _bin_edges(sample, bins=None, range=None):
    """ Create edge arrays
    """
    Dlen, Ndim = sample.shape
    nbin = np.empty(Ndim, int)
    edges = Ndim * [None]
    dedges = Ndim * [None]
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        if len(range) != Ndim:
            raise ValueError(f'range given for {len(range)} dimensions; {Ndim} required')
        smin = np.empty(Ndim)
        smax = np.empty(Ndim)
        for i in builtins.range(Ndim):
            if range[i][1] < range[i][0]:
                raise ValueError('In {}range, start must be <= stop'.format(f'dimension {i + 1} of ' if Ndim > 1 else ''))
            smin[i], smax[i] = range[i]
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - 0.5
            smax[i] = smax[i] + 0.5
    edges_dtype = sample.dtype if np.issubdtype(sample.dtype, np.floating) else float
    for i in builtins.range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1, dtype=edges_dtype)
        else:
            edges[i] = np.asarray(bins[i], edges_dtype)
            nbin[i] = len(edges[i]) + 1
        dedges[i] = np.diff(edges[i])
    nbin = np.asarray(nbin)
    return (nbin, edges, dedges)