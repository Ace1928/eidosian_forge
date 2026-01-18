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
def _set_graph_params(self, **params):
    if self.graph is not None:
        try:
            if 'n_pca' in params:
                params['n_pca'] = self._parse_n_pca(self.graph.data_nu, params['n_pca'])
            if 'n_svd' in params:
                params['n_svd'] = self._parse_n_svd(self.graph.data_nu, params['n_svd'])
            if 'n_landmark' in params:
                params['n_landmark'] = self._parse_n_landmark(self.graph.data_nu, params['n_landmark'])
            self.graph.set_params(**params)
        except ValueError as e:
            _logger.log_debug('Reset graph due to {}'.format(str(e)))
            self.graph = None