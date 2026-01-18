import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple
import torch
import torch.nn as nn
def _patched_create_proxy(self, create_proxy: Callable, exec_info: _ExecutionInfo, fqn_to_param: Dict[str, nn.Parameter], kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], name: Optional[str]=None, type_expr: Optional[Any]=None, proxy_factory_fn: Optional[Callable[[torch.fx.Node], torch.fx.Proxy]]=None) -> torch.fx.Proxy:
    """
        Overrides ``create_proxy`` to save execution information to
        ``exec_info``. Note that ``create_proxy`` is called during symbolic
        tracing for each leaf function/method/module.

        Args:
            create_proxy (Callable): Original ``create_proxy`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            fqn_to_param (Dict[str, nn.Parameter]): ``dict`` version of the
                root module's ``named_parameters()`` with FQN as key and
                parameter as value.
            kind (str): Kind of the target method ('call_function',
                'call_method', 'get_attr', 'call_module', 'placeholder', or
                'output'). See :class:`torch.fx.Graph` for details. This is
                passed to ``create_proxy``.
            target (torch.fx.node.Target): Contains the string name of the
                function/method/module. This is passed to ``create_proxy``.
            args (Tuple[Any, ...]): Positional arguments for the function/
                method/module. This is passed to ``create_proxy``.
            kwargs (Dict[str, Any]): Keyword arguments for the function/method/
                module. This is passed to ``create_proxy``
            name (Optional[str]): An optional string name for the ``Node``
                created in ``create_proxy``. This is passed to
                ``create_proxy``.
            type_expr (Optional[Any]): An optional type annotation representing
                the Python type that the output of the node has. This is passed
                to ``create_proxy``.
            proxy_factory_fn (Callable[[torch.fx.Node], torch.fx.Proxy]):
                An alternative proxy constructor used in ``create_proxy``. This
                is passed to ``create_proxy``.

        Returns:
            torch.fx.Proxy: Created ``Node`` wrapped in a ``Proxy`` object.
        """
    proxy = create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
    curr_module = exec_info.curr_module
    if kind in ('call_function', 'call_method'):
        if args is not None:
            named_params: List[Tuple[str, nn.Parameter]] = []
            for arg in args:
                if isinstance(arg, torch.fx.Proxy) and arg.node.target in fqn_to_param:
                    param = fqn_to_param[arg.node.target]
                    named_params.append((arg.node.target, param))
                    if param not in exec_info.visited_params:
                        exec_info.visited_params.add(param)
                        exec_info.param_forward_order.append(param)
            if named_params:
                exec_info.module_to_param_usage_infos[curr_module].append(_ParamUsageInfo(curr_module, named_params))
    elif kind == 'call_module':
        named_params = list(curr_module.named_parameters())
        if named_params:
            exec_info.module_to_param_usage_infos[curr_module].append(_ParamUsageInfo(curr_module, named_params))
        for _, param in named_params:
            if param not in exec_info.visited_params:
                exec_info.visited_params.add(param)
                exec_info.param_forward_order.append(param)
    return proxy