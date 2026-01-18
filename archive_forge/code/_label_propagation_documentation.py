import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
Graph matrix for Label Spreading computes the graph laplacian