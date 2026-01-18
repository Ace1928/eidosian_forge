from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union
import torch
from torch._C import DisableTorchFunctionSubclass
from torch.types import _device, _dtype, _size
from torchvision.tv_tensors._torch_function_helpers import _FORCE_TORCHFUNCTION_SUBCLASS, _must_return_subclass
class TVTensor(torch.Tensor):
    """[Beta] Base class for all TVTensors.

    You probably don't want to use this class unless you're defining your own
    custom TVTensors. See
    :ref:`sphx_glr_auto_examples_transforms_plot_custom_tv_tensors.py` for details.
    """

    @staticmethod
    def _to_tensor(data: Any, dtype: Optional[torch.dtype]=None, device: Optional[Union[torch.device, str, int]]=None, requires_grad: Optional[bool]=None) -> torch.Tensor:
        if requires_grad is None:
            requires_grad = data.requires_grad if isinstance(data, torch.Tensor) else False
        return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)

    @classmethod
    def _wrap_output(cls, output: torch.Tensor, args: Sequence[Any]=(), kwargs: Optional[Mapping[str, Any]]=None) -> torch.Tensor:
        if isinstance(output, torch.Tensor) and (not isinstance(output, cls)):
            output = output.as_subclass(cls)
        if isinstance(output, (tuple, list)):
            output = type(output)((cls._wrap_output(part, args, kwargs) for part in output))
        return output

    @classmethod
    def __torch_function__(cls, func: Callable[..., torch.Tensor], types: Tuple[Type[torch.Tensor], ...], args: Sequence[Any]=(), kwargs: Optional[Mapping[str, Any]]=None) -> torch.Tensor:
        """For general information about how the __torch_function__ protocol works,
        see https://pytorch.org/docs/stable/notes/extending.html#extending-torch

        TL;DR: Every time a PyTorch operator is called, it goes through the inputs and looks for the
        ``__torch_function__`` method. If one is found, it is invoked with the operator as ``func`` as well as the
        ``args`` and ``kwargs`` of the original call.

        Why do we override this? Because the base implementation in torch.Tensor would preserve the TVTensor type
        of the output. In our case, we want to return pure tensors instead (with a few exceptions). Refer to the
        "TVTensors FAQ" gallery example for a rationale of this behaviour (TL;DR: perf + no silver bullet).

        Our implementation below is very similar to the base implementation in ``torch.Tensor`` - go check it out.
        """
        if not all((issubclass(cls, t) for t in types)):
            return NotImplemented
        with DisableTorchFunctionSubclass():
            output = func(*args, **kwargs or dict())
        must_return_subclass = _must_return_subclass()
        if must_return_subclass or (func in _FORCE_TORCHFUNCTION_SUBCLASS and isinstance(args[0], cls)):
            return cls._wrap_output(output, args, kwargs)
        if not must_return_subclass and isinstance(output, cls):
            return output.as_subclass(torch.Tensor)
        return output

    def _make_repr(self, **kwargs: Any) -> str:
        extra_repr = ', '.join((f'{key}={value}' for key, value in kwargs.items()))
        return f'{super().__repr__()[:-1]}, {extra_repr})'

    @property
    def shape(self) -> _size:
        with DisableTorchFunctionSubclass():
            return super().shape

    @property
    def ndim(self) -> int:
        with DisableTorchFunctionSubclass():
            return super().ndim

    @property
    def device(self, *args: Any, **kwargs: Any) -> _device:
        with DisableTorchFunctionSubclass():
            return super().device

    @property
    def dtype(self) -> _dtype:
        with DisableTorchFunctionSubclass():
            return super().dtype

    def __deepcopy__(self: D, memo: Dict[int, Any]) -> D:
        return self.detach().clone().requires_grad_(self.requires_grad)