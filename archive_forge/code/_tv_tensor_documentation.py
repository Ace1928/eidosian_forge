from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union
import torch
from torch._C import DisableTorchFunctionSubclass
from torch.types import _device, _dtype, _size
from torchvision.tv_tensors._torch_function_helpers import _FORCE_TORCHFUNCTION_SUBCLASS, _must_return_subclass
For general information about how the __torch_function__ protocol works,
        see https://pytorch.org/docs/stable/notes/extending.html#extending-torch

        TL;DR: Every time a PyTorch operator is called, it goes through the inputs and looks for the
        ``__torch_function__`` method. If one is found, it is invoked with the operator as ``func`` as well as the
        ``args`` and ``kwargs`` of the original call.

        Why do we override this? Because the base implementation in torch.Tensor would preserve the TVTensor type
        of the output. In our case, we want to return pure tensors instead (with a few exceptions). Refer to the
        "TVTensors FAQ" gallery example for a rationale of this behaviour (TL;DR: perf + no silver bullet).

        Our implementation below is very similar to the base implementation in ``torch.Tensor`` - go check it out.
        