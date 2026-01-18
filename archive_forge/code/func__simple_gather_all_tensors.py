from typing import Any, List, Optional
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result