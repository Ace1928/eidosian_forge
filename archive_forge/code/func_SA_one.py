import types, math, os, gc
import torch
from torch.nn import functional as F
@MyFunction
def SA_one(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
    xx = state[5 * i + 1].to(dtype=self.FLOAT_MODE)
    xk = x * time_mix_k + xx * (1 - time_mix_k)
    xv = x * time_mix_v + xx * (1 - time_mix_v)
    xr = x * time_mix_r + xx * (1 - time_mix_r)
    state[5 * i + 1] = x.float()
    r = torch.sigmoid(xr @ rw)
    k = (xk @ kw).float()
    v = (xv @ vw).float()
    aa = state[5 * i + 2]
    bb = state[5 * i + 3]
    pp = state[5 * i + 4]
    ww = time_first + k
    p = torch.maximum(pp, ww)
    e1 = torch.exp(pp - p)
    e2 = torch.exp(ww - p)
    a = e1 * aa + e2 * v
    b = e1 * bb + e2
    ww = pp + time_decay
    p = torch.maximum(ww, k)
    e1 = torch.exp(ww - p)
    e2 = torch.exp(k - p)
    state[5 * i + 2] = e1 * aa + e2 * v
    state[5 * i + 3] = e1 * bb + e2
    state[5 * i + 4] = p
    wkv = (a / b).to(dtype=self.FLOAT_MODE)
    return r * wkv @ ow