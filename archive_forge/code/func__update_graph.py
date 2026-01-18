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
def _update_graph(self, X, precomputed, n_pca, n_landmark, **kwargs):
    if self.X is not None and (not matrix.matrix_is_equivalent(X, self.X)):
        '\n            If the same data is used, we can reuse existing kernel and\n            diffusion matrices. Otherwise we have to recompute.\n            '
        self.graph = None
    else:
        self._update_n_landmark(n_landmark)
        self._set_graph_params(n_pca=n_pca, precomputed=precomputed, n_landmark=n_landmark, random_state=self.random_state, knn=self.knn, decay=self.decay, distance=self.distance, n_svd=self._parse_n_svd(self.X, self.n_svd), n_jobs=self.n_jobs, thresh=self.thresh, verbose=self.verbose, **self.kwargs)
        if self.graph is not None:
            _logger.log_info('Using precomputed graph and diffusion operator...')