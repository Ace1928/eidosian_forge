import copy
import numbers
from abc import ABCMeta, abstractmethod
from math import ceil
from numbers import Integral, Real
import numpy as np
from scipy.sparse import issparse
from ..base import (
from ..utils import Bunch, check_random_state, compute_sample_weight
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import (
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import (
from ._utils import _any_isnan_axis0
def _prune_tree(self):
    """Prune tree using Minimal Cost-Complexity Pruning."""
    check_is_fitted(self)
    if self.ccp_alpha == 0.0:
        return
    if is_classifier(self):
        n_classes = np.atleast_1d(self.n_classes_)
        pruned_tree = Tree(self.n_features_in_, n_classes, self.n_outputs_)
    else:
        pruned_tree = Tree(self.n_features_in_, np.array([1] * self.n_outputs_, dtype=np.intp), self.n_outputs_)
    _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)
    self.tree_ = pruned_tree