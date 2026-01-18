from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad
from typing_extensions import TypeAlias
Return the device type list that supports fused kernels in optimizer.