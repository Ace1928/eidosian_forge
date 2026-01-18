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
@register_kl(Normal, Laplace)
def _kl_normal_laplace(p, q):
    loc_diff = p.loc - q.loc
    scale_ratio = p.scale / q.scale
    loc_diff_scale_ratio = loc_diff / p.scale
    t1 = torch.log(scale_ratio)
    t2 = math.sqrt(2 / math.pi) * p.scale * torch.exp(-0.5 * loc_diff_scale_ratio.pow(2))
    t3 = loc_diff * torch.erf(math.sqrt(0.5) * loc_diff_scale_ratio)
    return -t1 + (t2 + t3) / q.scale - 0.5 * (1 + math.log(0.5 * math.pi))