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
class FuseBatchNorm1dInLinear(Transformation):
    """
    Transformation that fuses `nn.BatchNorm1d` following or preceding `nn.Linear` into a single `nn.Linear`.
    The fusion will be done only if the linear layer has the batch normalization as sole following node, or the batch normalization
    has the linear layer as sole following node.

    For example, fusion will not be done in the case
    ```
         Linear
         /   \\
        /     \\
    ReLU   BatchNorm1d
    ```

    Example:
    ```python
    >>> from transformers.utils.fx import symbolic_trace
    >>> from transformers import AutoModel

    >>> from optimum.fx.optimization import FuseBatchNorm1dInLinear

    >>> model = AutoModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
    >>> model.eval()  # doctest: +IGNORE_RESULT

    >>> traced_model = symbolic_trace(
    ...     model,
    ...     input_names=["input_ids", "attention_mask", "pixel_values"],
    ...     disable_check=True
    ... )

    >>> transformation = FuseBatchNorm1dInLinear()
    >>> transformed_model = transformation(traced_model)
    ```
    """
    preserves_computation = True

    def transform(self, graph_module: 'GraphModule') -> 'GraphModule':
        for node in graph_module.graph.nodes:
            if node.op == 'call_module' and node.args[0].op == 'call_module':
                if type(graph_module.get_submodule(node.target)) is torch.nn.BatchNorm1d and type(graph_module.get_submodule(node.args[0].target)) is torch.nn.Linear:
                    if len(node.args[0].users) > 1:
                        continue
                    candidate_linear = graph_module.get_submodule(node.args[0].target)
                    candidate_batchnorm1d = graph_module.get_submodule(node.target)
                    if candidate_linear.weight.shape[0] == candidate_batchnorm1d.weight.shape[0]:
                        fused_linear = self.fuse(linear=candidate_linear, bn1d=candidate_batchnorm1d, bn1d_before=False)
                        parent_name, _, name = node.args[0].target.rpartition('.')
                        parent_module = graph_module.get_submodule(parent_name)
                        setattr(parent_module, name, fused_linear)
                        parent_name, _, name = node.target.rpartition('.')
                        parent_module = graph_module.get_submodule(parent_name)
                        delattr(parent_module, name)
                        node.replace_all_uses_with(node.args[0])
                        graph_module.graph.erase_node(node)
                elif type(graph_module.get_submodule(node.target)) is torch.nn.Linear and type(graph_module.get_submodule(node.args[0].target)) is torch.nn.BatchNorm1d:
                    if len(node.args[0].users) > 1:
                        continue
                    candidate_linear = graph_module.get_submodule(node.target)
                    candidate_batchnorm1d = graph_module.get_submodule(node.args[0].target)
                    if candidate_batchnorm1d.weight.shape[0] == candidate_linear.weight.shape[1]:
                        fused_linear = self.fuse(linear=candidate_linear, bn1d=candidate_batchnorm1d, bn1d_before=True)
                        parent_name, _, name = node.target.rpartition('.')
                        parent_module = graph_module.get_submodule(parent_name)
                        setattr(parent_module, name, fused_linear)
                        parent_name, _, name = node.args[0].target.rpartition('.')
                        parent_module = graph_module.get_submodule(parent_name)
                        delattr(parent_module, name)
                        batchnorm_node = node.args[0]
                        node.args[0].replace_all_uses_with(node.args[0].args[0])
                        graph_module.graph.erase_node(batchnorm_node)
        return graph_module

    def fuse(self, linear: torch.nn.Linear, bn1d: torch.nn.BatchNorm1d, bn1d_before: bool):
        linear_b = linear.bias if linear.bias is not None else torch.zeros_like(bn1d.running_mean)
        bn_w = bn1d.weight if bn1d.weight is not None else torch.ones_like(bn1d.running_mean)
        bn_b = bn1d.bias if bn1d.bias is not None else torch.ones_like(bn1d.running_mean)
        bn_var_rsqrt = torch.rsqrt(bn1d.running_var + bn1d.eps)
        if bn1d_before:
            linear.bias = torch.nn.Parameter(linear.weight @ (-bn_w * bn1d.running_mean * bn_var_rsqrt + bn_b) + linear_b)
            linear.weight = torch.nn.Parameter(linear.weight * (bn_w * bn_var_rsqrt)[None, :])
        else:
            linear.bias = torch.nn.Parameter((linear_b - bn1d.running_mean) * bn_var_rsqrt * bn_w + bn_b)
            linear.weight = torch.nn.Parameter(linear.weight * (bn_w * bn_var_rsqrt)[:, None])
        return linear