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
def _update_n_landmark(self, n_landmark):
    if self.graph is not None:
        n_landmark = self._parse_n_landmark(self.graph.data_nu, n_landmark)
        if n_landmark is None and isinstance(self.graph, graphs.LandmarkGraph) or (n_landmark is not None and (not isinstance(self.graph, graphs.LandmarkGraph))):
            kernel = self.graph.kernel
            self.graph = None
            self.fit(self.X, initialize=False)
            self.graph._kernel = kernel