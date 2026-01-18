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
@register_kl(MultivariateNormal, LowRankMultivariateNormal)
def _kl_multivariatenormal_lowrankmultivariatenormal(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError('KL-divergence between two (Low Rank) Multivariate Normals with                          different event shapes cannot be computed')
    term1 = _batch_lowrank_logdet(q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q._capacitance_tril) - 2 * p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    term3 = _batch_lowrank_mahalanobis(q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q.loc - p.loc, q._capacitance_tril)
    qWt_qDinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    A = torch.linalg.solve_triangular(q._capacitance_tril, qWt_qDinv, upper=False)
    term21 = _batch_trace_XXT(p._unbroadcasted_scale_tril * q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1))
    term22 = _batch_trace_XXT(A.matmul(p._unbroadcasted_scale_tril))
    term2 = term21 - term22
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])