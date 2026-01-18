import math
import warnings
from functools import total_ordering
from typing import Callable, Dict, Tuple, Type
import torch
from torch import inf
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_normal import HalfNormal
from .independent import Independent
from .laplace import Laplace
from .lowrank_multivariate_normal import (
from .multivariate_normal import _batch_mahalanobis, MultivariateNormal
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform
from .utils import _sum_rightmost, euler_constant as _euler_gamma
def _dispatch_kl(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [(super_p, super_q) for super_p, super_q in _KL_REGISTRY if issubclass(type_p, super_p) and issubclass(type_q, super_q)]
    if not matches:
        return NotImplemented
    left_p, left_q = min((_Match(*m) for m in matches)).types
    right_q, right_p = min((_Match(*reversed(m)) for m in matches)).types
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn('Ambiguous kl_divergence({}, {}). Please register_kl({}, {})'.format(type_p.__name__, type_q.__name__, left_p.__name__, right_q.__name__), RuntimeWarning)
    return left_fun