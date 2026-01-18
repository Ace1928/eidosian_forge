from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class PrependParamsAndBuffersAotAutogradOutputStep(OutputAdaptStep):
    """Prepend model's mutated buffers to the user output.

    :func:`torch.export.export` lifts model's mutated buffers as outputs, thus, they
    must be added to the user output after the model is executed.

    Args:
        model: The PyTorch model with mutated buffers.
    """

    def apply(self, model_outputs: Any, model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Sequence[Any]:
        """Flatten the model outputs and validate the `SpecTree` output.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            flattened_outputs: The flattened model outputs.
        """
        assert isinstance(model, torch_export.ExportedProgram), "'model' must be torch_export.ExportedProgram"
        ordered_buffers = tuple((model.state_dict[name] for name in model.graph_signature.buffers_to_mutate.values()))
        updated_outputs = (*ordered_buffers, *model_outputs)
        return updated_outputs