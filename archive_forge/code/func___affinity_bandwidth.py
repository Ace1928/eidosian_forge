from decorator import decorator
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import sklearn
import sklearn.cluster
import sklearn.feature_extraction
import sklearn.neighbors
from ._cache import cache
from . import util
from .filters import diagonal_filter
from .util.exceptions import ParameterError
from typing import Any, Callable, Optional, TypeVar, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
def __affinity_bandwidth(rec: scipy.sparse.csr_matrix, bw_mode: Optional[Union[np.ndarray, _FloatLike_co, str]], k: int) -> Union[float, np.ndarray]:
    if isinstance(bw_mode, np.ndarray):
        bandwidth = bw_mode
        if bandwidth.shape != rec.shape:
            raise ParameterError(f'Invalid matrix bandwidth shape: {bandwidth.shape}.Should be {rec.shape}.')
        if (bandwidth <= 0).any():
            raise ParameterError('Invalid bandwidth. All entries must be strictly positive.')
        return np.array(bandwidth[rec.nonzero()])
    elif isinstance(bw_mode, (int, float)):
        scalar_bandwidth = float(bw_mode)
        if scalar_bandwidth <= 0:
            raise ParameterError(f'Invalid scalar bandwidth={scalar_bandwidth}. Must be strictly positive.')
        return scalar_bandwidth
    if bw_mode is None:
        bw_mode = 'med_k_scalar'
    if bw_mode not in ['med_k_scalar', 'mean_k', 'gmean_k', 'mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair']:
        raise ParameterError(f"Invalid bandwidth='{bw_mode}'. Must be either a positive scalar or one of ['med_k_scalar', 'mean_k', 'gmean_k', 'mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair']")
    t = rec.shape[0]
    knn_dists = []
    for i in range(t):
        links = rec[i].nonzero()[1]
        if len(links) == 0:
            if bw_mode not in ['med_k_scalar']:
                raise ParameterError(f'The sample at time point {i} has no neighbors')
            else:
                knn_dists.append(np.array([np.nan]))
        else:
            knn_dist_row = np.sort(rec[i, links].toarray()[0])[:k]
            knn_dists.append(knn_dist_row)
    dist_to_k = np.asarray([dists[-1] for dists in knn_dists])
    avg_dist_to_first_ks = np.asarray([np.mean(dists) for dists in knn_dists])
    if bw_mode == 'med_k_scalar':
        if not np.any(np.isfinite(dist_to_k)):
            raise ParameterError('Cannot estimate bandwidth from an empty graph')
        return float(np.nanmedian(dist_to_k))
    if bw_mode in ['mean_k', 'gmean_k']:
        sigma_i_data = np.empty_like(rec.data)
        sigma_j_data = np.empty_like(rec.data)
        for row in range(t):
            sigma_i_data[rec.indptr[row]:rec.indptr[row + 1]] = dist_to_k[row]
            col_idx = rec.indices[rec.indptr[row]:rec.indptr[row + 1]]
            sigma_j_data[rec.indptr[row]:rec.indptr[row + 1]] = dist_to_k[col_idx]
        if bw_mode == 'mean_k':
            out = np.array((sigma_i_data + sigma_j_data) / 2)
        elif bw_mode == 'gmean_k':
            out = np.array((sigma_i_data * sigma_j_data) ** 0.5)
    if bw_mode in ['mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair']:
        sigma_i_data = np.empty_like(rec.data)
        sigma_j_data = np.empty_like(rec.data)
        for row in range(t):
            sigma_i_data[rec.indptr[row]:rec.indptr[row + 1]] = avg_dist_to_first_ks[row]
            col_idx = rec.indices[rec.indptr[row]:rec.indptr[row + 1]]
            sigma_j_data[rec.indptr[row]:rec.indptr[row + 1]] = avg_dist_to_first_ks[col_idx]
        if bw_mode == 'mean_k_avg':
            out = np.array((sigma_i_data + sigma_j_data) / 2)
        elif bw_mode == 'gmean_k_avg':
            out = np.array((sigma_i_data * sigma_j_data) ** 0.5)
        elif bw_mode == 'mean_k_avg_and_pair':
            out = np.array((sigma_i_data + sigma_j_data + rec.data) / 3)
    return out