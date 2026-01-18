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
@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p, q):
    ct1 = p.scale / q.scale
    ct2 = q.loc / q.scale
    ct3 = p.loc / q.scale
    t1 = -ct1.log() - ct2 + ct3
    t2 = ct1 * _euler_gamma
    t3 = torch.exp(ct2 + (1 + ct1).lgamma() - ct3)
    return t1 + t2 + t3 - (1 + _euler_gamma)