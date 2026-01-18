import numbers
from heapq import heappop, heappush
from timeit import default_timer as time
import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from ._bitset import set_raw_bitset_from_binned_bitset
from .common import (
from .histogram import HistogramBuilder
from .predictor import TreePredictor
from .splitting import Splitter
from .utils import sum_parallel
def _fill_predictor_arrays(predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets, grower_node, binning_thresholds, n_bins_non_missing, next_free_node_idx=0, next_free_bitset_idx=0):
    """Helper used in make_predictor to set the TreePredictor fields."""
    node = predictor_nodes[next_free_node_idx]
    node['count'] = grower_node.n_samples
    node['depth'] = grower_node.depth
    if grower_node.split_info is not None:
        node['gain'] = grower_node.split_info.gain
    else:
        node['gain'] = -1
    node['value'] = grower_node.value
    if grower_node.is_leaf:
        node['is_leaf'] = True
        return (next_free_node_idx + 1, next_free_bitset_idx)
    split_info = grower_node.split_info
    feature_idx, bin_idx = (split_info.feature_idx, split_info.bin_idx)
    node['feature_idx'] = feature_idx
    node['bin_threshold'] = bin_idx
    node['missing_go_to_left'] = split_info.missing_go_to_left
    node['is_categorical'] = split_info.is_categorical
    if split_info.bin_idx == n_bins_non_missing[feature_idx] - 1:
        node['num_threshold'] = np.inf
    elif split_info.is_categorical:
        categories = binning_thresholds[feature_idx]
        node['bitset_idx'] = next_free_bitset_idx
        binned_left_cat_bitsets[next_free_bitset_idx] = split_info.left_cat_bitset
        set_raw_bitset_from_binned_bitset(raw_left_cat_bitsets[next_free_bitset_idx], split_info.left_cat_bitset, categories)
        next_free_bitset_idx += 1
    else:
        node['num_threshold'] = binning_thresholds[feature_idx][bin_idx]
    next_free_node_idx += 1
    node['left'] = next_free_node_idx
    next_free_node_idx, next_free_bitset_idx = _fill_predictor_arrays(predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets, grower_node.left_child, binning_thresholds=binning_thresholds, n_bins_non_missing=n_bins_non_missing, next_free_node_idx=next_free_node_idx, next_free_bitset_idx=next_free_bitset_idx)
    node['right'] = next_free_node_idx
    return _fill_predictor_arrays(predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets, grower_node.right_child, binning_thresholds=binning_thresholds, n_bins_non_missing=n_bins_non_missing, next_free_node_idx=next_free_node_idx, next_free_bitset_idx=next_free_bitset_idx)