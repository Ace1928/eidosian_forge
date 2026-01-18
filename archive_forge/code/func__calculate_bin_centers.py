from typing import Dict, Optional, Tuple
import torch
def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers