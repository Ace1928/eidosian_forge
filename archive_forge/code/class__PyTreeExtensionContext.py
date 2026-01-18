from __future__ import annotations
import contextlib
import functools
import inspect
from typing import (
import torch._dynamo
import torch.export as torch_export
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.utils import _pytree as pytree
class _PyTreeExtensionContext:
    """Context manager to register PyTree extension."""
    _extensions: Dict[Type, Tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]

    def __init__(self):
        self._extensions = {}
        self._register_huggingface_model_output_extension()

    def __enter__(self):
        for class_type, (flatten_func, unflatten_func) in self._extensions.items():
            pytree._private_register_pytree_node(class_type, flatten_func, unflatten_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for class_type in self._extensions:
            pytree.SUPPORTED_NODES.pop(class_type)

    @_beartype.beartype
    def register_pytree_node(self, class_type: Type, flatten_func: pytree.FlattenFunc, unflatten_func: pytree.UnflattenFunc):
        """Register PyTree extension for a custom python type.

        Args:
            class_type: The custom python type.
            flatten_func: The flatten function.
            unflatten_func: The unflatten function.

        Raises:
            AssertionError: If the custom python type is already registered.
        """
        if class_type in pytree.SUPPORTED_NODES or class_type in self._extensions:
            return
        self._extensions[class_type] = (flatten_func, unflatten_func)

    def _register_huggingface_model_output_extension(self):
        try:
            from transformers import modeling_outputs
        except ImportError as e:
            return

        @_beartype.beartype
        def model_output_flatten(output: modeling_outputs.ModelOutput) -> Tuple[List[Any], pytree.Context]:
            return (list(output.values()), (type(output), list(output.keys())))

        @_beartype.beartype
        def model_output_unflatten(values: List[Any], context: pytree.Context) -> modeling_outputs.ModelOutput:
            output_type, keys = context
            return output_type(**dict(zip(keys, values)))
        named_model_output_classes = inspect.getmembers(modeling_outputs, lambda x: inspect.isclass(x) and issubclass(x, modeling_outputs.ModelOutput) and (x is not modeling_outputs.ModelOutput))
        for _, class_type in named_model_output_classes:
            self.register_pytree_node(class_type, model_output_flatten, model_output_unflatten)