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
@abc.abstractmethod
def _reset_graph(self):
    """Trigger a reset of self.graph

        Any downstream effects of resetting the graph should override this function
        """
    raise NotImplementedError