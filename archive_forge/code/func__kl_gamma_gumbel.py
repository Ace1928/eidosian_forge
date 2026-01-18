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
@register_kl(Gamma, Gumbel)
def _kl_gamma_gumbel(p, q):
    beta_scale_prod = p.rate * q.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = (p.concentration - 1) * p.concentration.digamma() - p.concentration.lgamma() - p.concentration
    t2 = beta_scale_prod.log() + p.concentration / beta_scale_prod
    t3 = torch.exp(loc_scale_ratio) * (1 + beta_scale_prod.reciprocal()).pow(-p.concentration) - loc_scale_ratio
    return t1 + t2 + t3