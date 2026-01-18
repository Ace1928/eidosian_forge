from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class FlattenOutputStep(OutputAdaptStep):
    """Flatten nested collection types and return a flat list of elements.

    ONNX can't represent collection types (e.g., dictionary, tuple of tuple of tensor,
    etc).

    NOTE: Ideally we would want to use ``FlattenOutputWithTreeSpecValidationOutputStep``, such
    that `SpecTree` can be validate for new model outputs. However, this is not possible
    currently because we never have access to real PyTorch model outputs during export.
    Only traced outputs may be available, but they are not an accurate reflection of the
    original PyTorch model outputs format as they are typically in their own unique format,
    depending on the tracing strategy.
    """

    def apply(self, model_outputs: Any, model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Sequence[Any]:
        """Flatten the model outputs.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            A tuple of the flattened model outputs.
        """
        return pytree.tree_leaves(model_outputs)