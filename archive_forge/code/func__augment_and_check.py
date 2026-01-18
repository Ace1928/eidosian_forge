import itertools as it
import random
import pytest
import networkx as nx
from networkx.algorithms.connectivity import k_edge_augmentation
from networkx.algorithms.connectivity.edge_augmentation import (
from networkx.utils import pairwise
def _augment_and_check(G, k, avail=None, weight=None, verbose=False, orig_k=None, max_aug_k=None):
    """
    Does one specific augmentation and checks for properties of the result
    """
    if orig_k is None:
        try:
            orig_k = nx.edge_connectivity(G)
        except nx.NetworkXPointlessConcept:
            orig_k = 0
    info = {}
    try:
        if avail is not None:
            avail_dict = dict(zip(*_unpack_available_edges(avail, weight=weight)))
        else:
            avail_dict = None
        try:
            generator = nx.k_edge_augmentation(G, k=k, weight=weight, avail=avail)
            assert not isinstance(generator, list), 'should always return an iter'
            aug_edges = []
            for edge in generator:
                aug_edges.append(edge)
        except nx.NetworkXUnfeasible:
            infeasible = True
            info['infeasible'] = True
            assert len(aug_edges) == 0, 'should not generate anything if unfeasible'
            if avail is None:
                n_nodes = G.number_of_nodes()
                assert n_nodes <= k, f'unconstrained cases are only unfeasible if |V| <= k. Got |V|={n_nodes} and k={k}'
            else:
                if max_aug_k is None:
                    G_aug_all = G.copy()
                    G_aug_all.add_edges_from(avail_dict.keys())
                    try:
                        max_aug_k = nx.edge_connectivity(G_aug_all)
                    except nx.NetworkXPointlessConcept:
                        max_aug_k = 0
                assert max_aug_k < k, 'avail should only be unfeasible if using all edges does not achieve k-edge-connectivity'
            partial_edges = list(nx.k_edge_augmentation(G, k=k, weight=weight, partial=True, avail=avail))
            info['n_partial_edges'] = len(partial_edges)
            if avail_dict is None:
                assert set(partial_edges) == set(complement_edges(G)), 'unweighted partial solutions should be the complement'
            elif len(avail_dict) > 0:
                H = G.copy()
                H.add_edges_from(partial_edges)
                partial_conn = nx.edge_connectivity(H)
                H.add_edges_from(set(avail_dict.keys()))
                full_conn = nx.edge_connectivity(H)
                assert partial_conn == full_conn, 'adding more edges should not increase k-conn'
            aug_edges = partial_edges
        else:
            infeasible = False
        num_edges = len(aug_edges)
        if avail is not None:
            total_weight = sum((avail_dict[e] for e in aug_edges))
        else:
            total_weight = num_edges
        info['total_weight'] = total_weight
        info['num_edges'] = num_edges
        G_aug = G.copy()
        G_aug.add_edges_from(aug_edges)
        try:
            aug_k = nx.edge_connectivity(G_aug)
        except nx.NetworkXPointlessConcept:
            aug_k = 0
        info['aug_k'] = aug_k
        if not infeasible and orig_k < k:
            assert info['aug_k'] >= k, f'connectivity should increase to k={k} or more'
        assert info['aug_k'] >= orig_k, 'augmenting should never reduce connectivity'
        _assert_solution_properties(G, aug_edges, avail_dict)
    except Exception:
        info['failed'] = True
        print(f'edges = {list(G.edges())}')
        print(f'nodes = {list(G.nodes())}')
        print(f'aug_edges = {list(aug_edges)}')
        print(f'info  = {info}')
        raise
    else:
        if verbose:
            print(f'info  = {info}')
    if infeasible:
        aug_edges = None
    return (aug_edges, info)