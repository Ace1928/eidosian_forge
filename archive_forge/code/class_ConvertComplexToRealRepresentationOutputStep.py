from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class ConvertComplexToRealRepresentationOutputStep(OutputAdaptStep):
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).

    """

    def apply(self, model_outputs: Any, model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Any:
        """Convert float tensors to complex tensors.

        Args:
            model_outputs: The model output.
            model: The PyTorch model.

        Returns:
            A tuple of the model output.
        """
        return [torch.view_as_real(output) if isinstance(output, torch.Tensor) and torch.is_complex(output) else output for output in model_outputs]