from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class InputAdapter:
    """A class that adapts the PyTorch model inputs to exported ONNX model inputs format."""

    def __init__(self, steps: Optional[List[InputAdaptStep]]=None):
        self._steps = steps or []

    @_beartype.beartype
    def append_step(self, step: InputAdaptStep) -> None:
        """Appends a step to the input adapt steps.

        Args:
            step: The step to append.
        """
        self._steps.append(step)

    @_beartype.beartype
    def apply(self, *model_args, model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None, **model_kwargs) -> Sequence[Union[int, float, bool, str, 'torch.Tensor', None]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Args:
            model_args: The PyTorch model inputs.
            model: The PyTorch model.
            model_kwargs: The PyTorch model keyword inputs.
        Returns:
            A sequence of tensors converted from PyTorch model inputs.
        """
        args: Sequence[Any] = model_args
        kwargs: Mapping[str, Any] = model_kwargs
        for step in self._steps:
            args, kwargs = step.apply(args, kwargs, model=model)
        assert not kwargs
        return args