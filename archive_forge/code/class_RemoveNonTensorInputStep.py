from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class RemoveNonTensorInputStep(InputAdaptStep):
    """Remove the non-tensor input arguments.

    Dynamo does not support non-tensor input arguments (https://github.com/pytorch/pytorch/issues/99534).

    Specifically, it does put the input into graph with an empty node, but consumed by no ones.
    The concrete value is embedded into the graph as a constant arg of a target node. Meta
    suggests in this case that one should rewrite the model code to make it tensor if the
    input value is supposed to change at runtime. We might need to further investigate
    the feasibility of that suggestion.

    For example,

        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        x = torch.randn(1, 1, 2, dtype=torch.float32)
        gm_fun, _ = dynamo.export(func, x, b=8.0, aten_graph=True, tracing_mode="real")

        # class GraphModule(torch.nn.Module):
        #     def forward(self, x, b):
        #         arg0: f32[1, 1, 2], arg1, = fx_pytree.tree_flatten_spec(([x, b], {}), self._in_spec)
        #         # File: path/to/pytorch/test_constant_input.py:5, code: y = x + b
        #         add_tensor: f32[1, 1, 2] = torch.ops.aten.add.Tensor(arg0, 8.0);  arg0 = None

        #         # File: path/to/pytorch/test_constant_input.py:6, code: z = y.relu()
        #         relu_default: f32[1, 1, 2] = torch.ops.aten.relu.default(add_tensor)
        #         return pytree.tree_unflatten([add_tensor, relu_default], self._out_spec)

    Empty torch.fx.Node input leading to a mismatched number of input with PyTorch, as
    it's ignored in ONNX graph. Thus, we delete the useless input here.

    """

    def apply(self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any], model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Remove Constant from arguments.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs.

        Raises:
            ValueError: If `model_kwargs` is not empty.
        """
        assert not model_kwargs
        return (tuple((arg for arg in model_args if not isinstance(arg, (int, float, bool, str)))), {})