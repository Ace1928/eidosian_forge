import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS
@compatibility(is_backward_compatible=False)
class OperatorSupport(OperatorSupportBase):
    """
    `_support_dict` maps node.target typename to supported inputs dtypes.

    node.target typename is retrieved using helper function `get_node_target()`

    If supported inputs dtypes is None, it means any dtype is supported, else
    we should see a tuple like (([dtypes], ...), {"name":[dtypes], ...}).

    The first tuple ([dtypes], ...) indicates what dtypes are supported for
    inputs in node.args and the second dict {"name": [dtypes], ...} indicates
    what dtypes are supported for inputs in node.kwargs.

    For inputs in args, if we don't want to check it, we can put None there,
    e.g. (None, [torch.float]) indicates that we don't care about the type of
    the first input in args. And for inputs in kwargs, if not listed, will not
    be checked.
    """
    _support_dict: SupportDict

    def __init__(self, support_dict: t.Optional[SupportDict]=None):
        self._support_dict = support_dict or {}

    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        """
        Args:
            `submodules`: mapping from module name to the module. This can be
                          retrieved by calling model.named_modules().

            `node`: a Fx node that we want to determine whether it's supported.

        Returns:
            `is_supported`: whether the arg `node` is supported.
        """
        if node.op not in CALLABLE_NODE_OPS:
            return True
        target = get_node_target(submodules, node)
        if target not in self._support_dict:
            return False
        if self._support_dict[target] is None:
            return True
        args_dtypes, kwargs_dtypes = self._support_dict[target]
        for i, dtypes in enumerate(args_dtypes):
            if len(node.args) <= i:
                break
            if dtypes is None:
                continue
            if not isinstance(node.args[i], torch.fx.Node):
                continue
            arg_dtype = _get_arg_dtype(node.args[i])
            if arg_dtype not in dtypes:
                return False
        for k, dtypes in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue
            kwarg_dtype = _get_arg_dtype(node.kwargs[k])
            if kwarg_dtype not in dtypes:
                return False
        return True