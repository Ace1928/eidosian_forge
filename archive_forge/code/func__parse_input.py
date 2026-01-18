from . import api
from . import base
from . import graphs
from . import matrix
from . import utils
from functools import partial
from scipy import sparse
import abc
import numpy as np
import pygsp
import tasklogger
def _parse_input(self, X):
    if isinstance(X, base.BaseGraph):
        self.graph = X
        X = X.data
        n_pca = self.graph.n_pca
        self.knn = self.graph.knn
        self.decay = self.graph.decay
        self.distance = self.graph.distance
        self.thresh = self.graph.thresh
        update_graph = False
        if isinstance(self.graph, graphs.TraditionalGraph):
            precomputed = self.graph.precomputed
        else:
            precomputed = None
    elif isinstance(X, pygsp.graphs.Graph):
        self.graph = None
        X = X.W
        precomputed = 'adjacency'
        update_graph = False
        n_pca = None
    else:
        update_graph = True
        if utils.is_Anndata(X):
            X = X.X
        if not callable(self.distance) and self.distance.startswith('precomputed'):
            if self.distance == 'precomputed':
                precomputed = self._detect_precomputed_matrix_type(X)
            elif self.distance in ['precomputed_affinity', 'precomputed_distance']:
                precomputed = self.distance.split('_')[1]
            else:
                raise NotImplementedError
            n_pca = None
        else:
            precomputed = None
            n_pca = self._parse_n_pca(X, self.n_pca)
    return (X, n_pca, self._parse_n_landmark(X, self.n_landmark), precomputed, update_graph)