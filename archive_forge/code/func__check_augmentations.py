import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def _check_augmentations(G, avail=None, max_k=None, weight=None, verbose=False):
    """Helper to check weighted/unweighted cases with multiple values of k"""
    try:
        orig_k = nx.edge_connectivity(G)
    except nx.NetworkXPointlessConcept:
        orig_k = 0
    if avail is not None:
        all_aug_edges = _unpack_available_edges(avail, weight=weight)[0]
        G_aug_all = G.copy()
        G_aug_all.add_edges_from(all_aug_edges)
        try:
            max_aug_k = nx.edge_connectivity(G_aug_all)
        except nx.NetworkXPointlessConcept:
            max_aug_k = 0
    else:
        max_aug_k = G.number_of_nodes() - 1
    if max_k is None:
        max_k = min(4, max_aug_k)
    avail_uniform = {e: 1 for e in complement_edges(G)}
    if verbose:
        print('\n=== CHECK_AUGMENTATION ===')
        print(f'G.number_of_nodes = {G.number_of_nodes()!r}')
        print(f'G.number_of_edges = {G.number_of_edges()!r}')
        print(f'max_k = {max_k!r}')
        print(f'max_aug_k = {max_aug_k!r}')
        print(f'orig_k = {orig_k!r}')
    for k in range(1, max_k + 1):
        if verbose:
            print('---------------')
            print(f'Checking k = {k}')
        if verbose:
            print('unweighted case')
        aug_edges1, info1 = _augment_and_check(G, k=k, verbose=verbose, orig_k=orig_k)
        if verbose:
            print('weighted uniform case')
        aug_edges2, info2 = _augment_and_check(G, k=k, avail=avail_uniform, verbose=verbose, orig_k=orig_k, max_aug_k=G.number_of_nodes() - 1)
        if avail is not None:
            if verbose:
                print('weighted case')
            aug_edges3, info3 = _augment_and_check(G, k=k, avail=avail, weight=weight, verbose=verbose, max_aug_k=max_aug_k, orig_k=orig_k)
        if aug_edges1 is not None:
            if k == 1:
                assert info2['total_weight'] == info1['total_weight']
            if k == 2:
                if orig_k == 0:
                    assert info2['total_weight'] <= info1['total_weight'] * 3
                else:
                    assert info2['total_weight'] <= info1['total_weight'] * 2
                _check_unconstrained_bridge_property(G, info1)