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
@register_kl(Normal, Gumbel)
def _kl_normal_gumbel(p, q):
    mean_scale_ratio = p.loc / q.scale
    var_scale_sqr_ratio = (p.scale / q.scale).pow(2)
    loc_scale_ratio = q.loc / q.scale
    t1 = var_scale_sqr_ratio.log() * 0.5
    t2 = mean_scale_ratio - loc_scale_ratio
    t3 = torch.exp(-mean_scale_ratio + 0.5 * var_scale_sqr_ratio + loc_scale_ratio)
    return -t1 + t2 + t3 - 0.5 * (1 + math.log(2 * math.pi))