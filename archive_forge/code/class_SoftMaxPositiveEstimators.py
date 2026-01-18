import math
from enum import Enum, auto
from typing import Optional
import torch
from torch.autograd.profiler import record_function
from .base import FeatureMap
class SoftMaxPositiveEstimators(FeatureMap):

    def __init__(self, dim_features: int, iter_before_redraw: Optional[int], normalize_inputs: bool=False, epsilon: float=1e-06, softmax_temp: float=-1):
        super().__init__(dim_features, iter_before_redraw, normalize_inputs, epsilon)
        self.softmax_temp = softmax_temp
        self.h_scale = math.log(math.sqrt(self.dim_features))

    def pre_scale(self, x: torch.Tensor) -> torch.Tensor:
        with record_function('feature_map::pre_scale'):
            if self.iter_before_redraw is not None and self._iter_counter > self.iter_before_redraw or self.features is None or self.features.device != x.device:
                self._iter_counter = 1
                self.features = self._get_feature_map(x.shape[-1], self.dim_feature_map, x.device)
            features = self.features
            assert features is not None
            if features.dtype != x.dtype:
                self.features = features.to(x.dtype)
            self._iter_counter += 1
            if self.softmax_temp < 0:
                self.softmax_temp = x.shape[-1] ** (-0.25)
            x_scaled = x * self.softmax_temp
            norm_x_2 = torch.einsum('...d,...d->...', x_scaled, x_scaled).unsqueeze(-1)
            self.offset = -0.5 * norm_x_2 - self.h_scale + self.epsilon
            if self.normalize_inputs:
                self.offset -= norm_x_2.max(1, keepdim=True)[0]
        return x_scaled

    @staticmethod
    @torch.no_grad()
    def _get_random_ortho_matrix(blocks: int, dim: int, device: torch.device, norm_distribution: NormDistribution=NormDistribution.Uniform) -> torch.Tensor:
        """
        Generate a random matrix whose rows are exactly orthonormal

        "How to generate random matrices from the classical compact groups", Mezzadri, 2007
        https://arxiv.org/pdf/math-ph/0609050v2.pdf

        .. note: the typical qr decomposition does not give uniform results, qr decomposition is not
        unique and the qr decomposition routines are biased towards numerical stability. See the above
        paper for more information.

        .. note: this does not follow the original implementation from the Performers authors.
        see docs/assets/kde plots to visualize the impact of using the R signs to correct Q
        """
        H = torch.randn((blocks, dim, dim), device=device, requires_grad=False)
        if norm_distribution == NormDistribution.Xi:
            norms = torch.sqrt(torch.einsum('...d,...d->...', H, H))
        Q, R = torch.linalg.qr(H)
        Q = torch.diag_embed(torch.sign(torch.diagonal(R, dim1=1, dim2=2))) @ Q
        if norm_distribution == NormDistribution.Xi:
            return torch.diag_embed(norms) @ Q
        return Q