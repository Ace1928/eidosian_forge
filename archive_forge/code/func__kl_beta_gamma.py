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
@register_kl(Beta, Gamma)
def _kl_beta_gamma(p, q):
    t1 = -p.entropy()
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (q.concentration - 1) * (p.concentration1.digamma() - (p.concentration1 + p.concentration0).digamma())
    t4 = q.rate * p.concentration1 / (p.concentration1 + p.concentration0)
    return t1 + t2 - t3 + t4