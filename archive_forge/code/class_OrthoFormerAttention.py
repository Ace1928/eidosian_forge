import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as Fn
from xformers.components.attention import (
from xformers.components.attention.core import (
@register_attention('orthoformer', OrthoformerAttentionConfig)
class OrthoFormerAttention(Attention):

    def __init__(self, dropout: float, num_landmarks: int=32, subsample_fraction: float=1.0, landmark_selection: LandmarkSelection=LandmarkSelection.Orthogonal, *args, **kwargs):
        """
        Orthoformer_ attention mechanism.
        ::

            "Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers"
            Patrick, M., Campbell, D., Asano, Y., Misra, I., Metze, F., Feichtenhofer,
            C., Vedaldi, A., Henriques, J. (2021)

            Reference codebase: https://github.com/facebookresearch/Motionformer

        .. _Orthoformer: https://arxiv.org/abs/2106.05392

        """
        super().__init__()
        self.num_landmarks = num_landmarks
        self.attn_drop = nn.Dropout(dropout)
        self.subsample_fraction = subsample_fraction
        self.landmark_selection = landmark_selection
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, att_mask: Optional[Union[AttentionMask, torch.Tensor]]=None, *args, **kwargs):
        N = k.shape[1]
        if self.num_landmarks == N:
            x = scaled_dot_product_attention(q, k, v, att_mask)
        else:
            with torch.no_grad(), profiler.record_function('select landmarks'):
                if self.landmark_selection == LandmarkSelection.Orthogonal:
                    landmarks = self._compute_orthogonal_landmarks(q)
                elif self.landmark_selection == LandmarkSelection.Random:
                    half_L = self.num_landmarks // 2
                    landmarks_q = q[:, torch.randint(q.size(1), (half_L,)), :]
                    landmarks_k = k[:, torch.randint(k.size(1), (half_L,)), :]
                    landmarks = torch.cat((landmarks_q, landmarks_k), dim=-2)
                elif self.landmark_selection == LandmarkSelection.KMeans:
                    landmarks = self._cluster_landmarks(q)
                elif self.landmark_selection == LandmarkSelection.KMeans_Spherical:
                    landmarks = self._cluster_landmarks(q, spherical=True)
            if att_mask is not None:
                logger.warning('Orthoformer: attention mask passed alongside with using landmarks to reduce dimensions.                     The two are typically not compatible')
                att_mask = None
            kernel_1 = scaled_query_key_softmax(q, landmarks, att_mask)
            kernel_2 = scaled_query_key_softmax(landmarks, k, att_mask)
            x = torch.matmul(kernel_1, torch.matmul(kernel_2, v))
        x = self.attn_drop(x)
        return x

    def _cluster_landmarks(self, q: torch.Tensor, spherical: bool=False, num_iters: int=6) -> torch.Tensor:
        """
        Construct set of landmarks by recursively selecting new landmarks
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """
        num_landmarks = min(self.num_landmarks, q.shape[1])
        if self.subsample_fraction < 1.0:
            num_samples = max(int(self.subsample_fraction * q.size(-2)), num_landmarks)
            q_samples = q[:, torch.randint(q.size(-2), (num_samples,)), :]
        else:
            q_samples = q
        if spherical:
            q_samples_normalized = Fn.normalize(q_samples, p=2, dim=-1)
            landmarks = self._kmeans_spherical(q_samples_normalized, num_landmarks, num_iters)
        else:
            landmarks = self._kmeans(q_samples, num_landmarks, num_iters)
        return landmarks

    def _kmeans(self, x: torch.Tensor, K: int, num_iters: int=10):
        """
        Arguments:
            x: (B, N, D)
            K: number of clusters
            num_iters: the number of kmeans updates
        """
        B, N, D = x.size()
        assert K <= N, f'{K} > {N}'
        c = x[:, torch.randperm(N, device=x.device)[:K], :].clone()
        with profiler.record_function('kmeans'):
            x_i = x.view(B, N, 1, D)
            c_j = c.view(B, 1, K, D)
            counts = c.new_zeros(B, K)
            ones = x.new_ones((B, N))
            for _ in range(num_iters):
                D_ij = ((x_i - c_j) ** 2).sum(-1)
                cl = D_ij.argmin(dim=-1, keepdim=True).long()
                c.zero_()
                c.scatter_add_(-2, cl.repeat(1, 1, D), x)
                counts.fill_(1e-06)
                counts.scatter_add_(-1, cl.squeeze(-1), ones)
                c.divide_(counts.unsqueeze(-1))
        return c

    def _kmeans_spherical(self, x: torch.Tensor, K: int, num_iters=10):
        """
        Arguments:
            x: (B, N, D)
        """
        B, N, D = x.size()
        assert K <= N, f'{K} > {N}'
        c = x[:, torch.randperm(N, device=x.device)[:K], :].clone()
        with profiler.record_function('kmeans_spherical'):
            counts = c.new_zeros(B, K)
            ones = x.new_ones((B, N))
            for _ in range(num_iters):
                D_ij = torch.matmul(x, c.transpose(-2, -1))
                cl = D_ij.argmax(dim=-1, keepdim=True).long()
                c.zero_()
                c.scatter_add_(-2, cl.repeat(1, 1, D), x)
                counts.fill_(1e-06)
                counts.scatter_add_(-1, cl.squeeze(-1), ones)
                c.divide_(counts.unsqueeze(-1))
                c = Fn.normalize(c, p=2, dim=-1)
        return c

    def _compute_orthogonal_landmarks(self, q: torch.Tensor) -> torch.Tensor:
        """
        Construct set of landmarks by recursively selecting new landmarks
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """
        if self.subsample_fraction < 1.0:
            num_samples = max(int(self.subsample_fraction * q.size(-2)), self.num_landmarks)
            q_samples = q[:, torch.randint(q.size(-2), (num_samples,), device=q.device), :]
        else:
            q_samples = q
        q_samples_normalized = Fn.normalize(q_samples, p=2, dim=-1)
        B, N, D = q_samples_normalized.shape
        selected_mask = torch.zeros((B, N, 1), device=q_samples_normalized.device)
        landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=q_samples_normalized.device)
        random_idx = torch.randint(q_samples_normalized.size(-2), (B, 1, 1), device=q_samples_normalized.device)
        selected_mask.scatter_(-2, random_idx, landmark_mask)
        selected_landmarks = torch.empty((B, self.num_landmarks, D), device=q_samples_normalized.device, dtype=q_samples_normalized.dtype)
        selected_landmarks[:, 0, :] = q_samples_normalized[torch.arange(q_samples_normalized.size(0)), random_idx.view(-1), :].view(B, D)
        cos_sims = torch.empty((B, N, self.num_landmarks), device=q_samples_normalized.device, dtype=q_samples_normalized.dtype)
        for M in range(1, self.num_landmarks):
            with profiler.record_function('find new landmark'):
                cos_sims[:, :, M - 1] = torch.einsum('b n d, b d -> b n', q_samples_normalized, selected_landmarks[:, M - 1, :]).abs()
                cos_sim_set = cos_sims[:, :, :M]
                cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
                selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
                selected_landmarks[:, M, :] = q_samples_normalized[torch.arange(q_samples_normalized.size(0)), selected_landmark_idx, :].view(B, D)
                selected_mask.scatter_(-2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask)
        landmarks = torch.masked_select(q_samples, selected_mask.bool()).reshape(B, -1, D)
        return landmarks