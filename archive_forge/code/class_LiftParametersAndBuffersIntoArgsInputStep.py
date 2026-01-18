from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class LiftParametersAndBuffersIntoArgsInputStep(InputAdaptStep):
    """Append parameters and buffers to model's positional argument list."""

    def __init__(self, inputs: Tuple['torch.Tensor', ...]) -> None:
        self.inputs = inputs

    def apply(self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any], model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Append model's parameters and buffers into its input.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args + appended inputs and kwargs.
        """
        return ((*model_args, *self.inputs), model_kwargs)