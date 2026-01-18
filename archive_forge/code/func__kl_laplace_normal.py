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
@register_kl(Laplace, Normal)
def _kl_laplace_normal(p, q):
    var_normal = q.scale.pow(2)
    scale_sqr_var_ratio = p.scale.pow(2) / var_normal
    t1 = 0.5 * torch.log(2 * scale_sqr_var_ratio / math.pi)
    t2 = 0.5 * p.loc.pow(2)
    t3 = p.loc * q.loc
    t4 = 0.5 * q.loc.pow(2)
    return -t1 + scale_sqr_var_ratio + (t2 - t3 + t4) / var_normal - 1