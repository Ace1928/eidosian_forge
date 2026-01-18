import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import torch.fx
import torch.nn as nn
from ...ops import MLP, StochasticDepth
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
def _add_rel_pos(attn: torch.Tensor, q: torch.Tensor, q_thw: Tuple[int, int, int], k_thw: Tuple[int, int, int], rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, rel_pos_t: torch.Tensor) -> torch.Tensor:
    q_t, q_h, q_w = q_thw
    k_t, k_h, k_w = k_thw
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    dt = int(2 * max(q_t, k_t) - 1)
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - (torch.arange(k_h)[None, :] + (1.0 - k_h)) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - (torch.arange(k_w)[None, :] + (1.0 - k_w)) * k_w_ratio
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = torch.arange(q_t)[:, None] * q_t_ratio - (torch.arange(k_t)[None, :] + (1.0 - k_t)) * k_t_ratio
    rel_pos_h = _interpolate(rel_pos_h, dh)
    rel_pos_w = _interpolate(rel_pos_w, dw)
    rel_pos_t = _interpolate(rel_pos_t, dt)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    Rt = rel_pos_t[dist_t.long()]
    B, n_head, _, dim = q.shape
    r_q = q[:, :, 1:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum('bythwc,hkc->bythwk', r_q, Rh)
    rel_w_q = torch.einsum('bythwc,wkc->bythwk', r_q, Rw)
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)
    rel_q_t = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    rel_q_t = rel_q_t.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)
    rel_pos = (rel_h_q[:, :, :, :, :, None, :, None] + rel_w_q[:, :, :, :, :, None, None, :] + rel_q_t[:, :, :, :, :, :, None, None]).reshape(B, n_head, q_t * q_h * q_w, k_t * k_h * k_w)
    attn[:, :, 1:, 1:] += rel_pos
    return attn