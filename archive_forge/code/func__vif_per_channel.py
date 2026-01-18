import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torchmetrics.utilities.distributed import reduce
def _vif_per_channel(preds: Tensor, target: Tensor, sigma_n_sq: float) -> Tensor:
    dtype = preds.dtype
    device = preds.device
    preds = preds.unsqueeze(1)
    target = target.unsqueeze(1)
    eps = torch.tensor(1e-10, dtype=dtype, device=device)
    sigma_n_sq = torch.tensor(sigma_n_sq, dtype=dtype, device=device)
    preds_vif, target_vif = (torch.zeros(1, dtype=dtype, device=device), torch.zeros(1, dtype=dtype, device=device))
    for scale in range(4):
        n = 2.0 ** (4 - scale) + 1
        kernel = _filter(n, n / 5, dtype=dtype, device=device)[None, None, :]
        if scale > 0:
            target = conv2d(target, kernel)[:, :, ::2, ::2]
            preds = conv2d(preds, kernel)[:, :, ::2, ::2]
        mu_target = conv2d(target, kernel)
        mu_preds = conv2d(preds, kernel)
        mu_target_sq = mu_target ** 2
        mu_preds_sq = mu_preds ** 2
        mu_target_preds = mu_target * mu_preds
        sigma_target_sq = torch.clamp(conv2d(target ** 2, kernel) - mu_target_sq, min=0.0)
        sigma_preds_sq = torch.clamp(conv2d(preds ** 2, kernel) - mu_preds_sq, min=0.0)
        sigma_target_preds = conv2d(target * preds, kernel) - mu_target_preds
        g = sigma_target_preds / (sigma_target_sq + eps)
        sigma_v_sq = sigma_preds_sq - g * sigma_target_preds
        mask = sigma_target_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        sigma_target_sq[mask] = 0
        mask = sigma_preds_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = 0
        mask = g < 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        g[mask] = 0
        sigma_v_sq = torch.clamp(sigma_v_sq, min=eps)
        preds_vif_scale = torch.log10(1.0 + g ** 2.0 * sigma_target_sq / (sigma_v_sq + sigma_n_sq))
        preds_vif = preds_vif + torch.sum(preds_vif_scale, dim=[1, 2, 3])
        target_vif = target_vif + torch.sum(torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3])
    return preds_vif / target_vif