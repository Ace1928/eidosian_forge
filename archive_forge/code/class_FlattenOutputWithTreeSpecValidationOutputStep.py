from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class FlattenOutputWithTreeSpecValidationOutputStep(OutputAdaptStep):
    """Same as ``FlattenOutputStep``, with additional `TreeSpec` validation.

    This class stores the `SpecTree` output produced when `adapt` was called the first
    time. It then validates the `SpecTree` output produced from later `adapt` calls.
    """
    _spec: Optional[pytree.TreeSpec] = None

    def apply(self, model_outputs: Any, model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Sequence[Any]:
        """Flatten the model outputs and validate the `SpecTree` output.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            flattened_outputs: The flattened model outputs.

        Raises:
            ValueError: If the `SpecTree` output produced from the current `model_outputs`
                is not identical to the `SpecTree` output produced from the first
                `model_outputs` that was passed to this method.
        """
        flattened_outputs, spec = pytree.tree_flatten(model_outputs)
        if self._spec is None:
            self._spec = spec
        else:
            _assert_identical_pytree_spec(self._spec, spec, error_message='Model outputs incompatible with the format that was exported. ')
        return flattened_outputs