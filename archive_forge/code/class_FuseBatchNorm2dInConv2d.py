import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
@add_end_docstrings(_ATTRIBUTES_DOCSTRING)
class FuseBatchNorm2dInConv2d(Transformation):
    """
    Transformation that fuses `nn.BatchNorm2d` following `nn.Conv2d` into a single `nn.Conv2d`.
    The fusion will be done only if the convolution has the batch normalization as sole following node.

    For example, fusion will not be done in the case
    ```
         Conv2d
         /   \\
        /     \\
    ReLU   BatchNorm2d
    ```

    Example:
    ```python
    >>> from transformers.utils.fx import symbolic_trace
    >>> from transformers import AutoModelForImageClassification

    >>> from optimum.fx.optimization import FuseBatchNorm2dInConv2d

    >>> model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    >>> model.eval()  # doctest: +IGNORE_RESULT

    >>> traced_model = symbolic_trace(
    ...     model,
    ...     input_names=["pixel_values"],
    ...     disable_check=True
    ... )

    >>> transformation = FuseBatchNorm2dInConv2d()
    >>> transformed_model = transformation(traced_model)
    ```
    """
    preserves_computation = True

    def transform(self, graph_module: 'GraphModule') -> 'GraphModule':
        for node in graph_module.graph.nodes:
            if node.op == 'call_module' and node.args[0].op == 'call_module':
                if type(graph_module.get_submodule(node.target)) is torch.nn.BatchNorm2d and type(graph_module.get_submodule(node.args[0].target)) is torch.nn.Conv2d:
                    if len(node.args[0].users) > 1:
                        continue
                    fused_conv = self.fuse(conv2d=graph_module.get_submodule(node.args[0].target), bn2d=graph_module.get_submodule(node.target))
                    parent_name, _, name = node.args[0].target.rpartition('.')
                    parent_module = graph_module.get_submodule(parent_name)
                    setattr(parent_module, name, fused_conv)
                    parent_name, _, name = node.target.rpartition('.')
                    parent_module = graph_module.get_submodule(parent_name)
                    delattr(parent_module, name)
                    node.replace_all_uses_with(node.args[0])
                    graph_module.graph.erase_node(node)
        return graph_module

    def fuse(self, conv2d: torch.nn.Conv2d, bn2d: torch.nn.BatchNorm2d):
        conv_b = conv2d.bias if conv2d.bias is not None else torch.zeros_like(bn2d.running_mean)
        bn_w = bn2d.weight if bn2d.weight is not None else torch.ones_like(bn2d.running_mean)
        bn_b = bn2d.bias if bn2d.bias is not None else torch.ones_like(bn2d.running_mean)
        bn_var_rsqrt = torch.rsqrt(bn2d.running_var + bn2d.eps)
        conv2d.weight = torch.nn.Parameter(conv2d.weight * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv2d.weight.shape) - 1)))
        conv2d.bias = torch.nn.Parameter(conv_b - bn2d.running_mean * bn_var_rsqrt * bn_w + bn_b)
        return conv2d