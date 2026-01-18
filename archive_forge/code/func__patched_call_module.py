import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple
import torch
import torch.nn as nn
def _patched_call_module(self, call_module: Callable, exec_info: _ExecutionInfo, module: nn.Module, forward: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """
        Overrides ``call_module`` to save execution information to
        ``exec_info``. Note that ``call_module`` is called during symbolic
        tracing for each non-root module.

        Args:
            call_module (Callable): Original ``call_module`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            module (nn.Module): Module corresponding to this ``call_module``.
            forward (Callable): ``forward()`` method of ``module`` to be called
                for this ``call_module``.
            args (Tuple[Any, ...]): Positional arguments for ``forward``.
            kwargs (Dict[str, Any]): Keyword arguments for ``forward``.

        Returns:
            Same return value as ``call_module``.
        """
    exec_info.module_forward_order.append(module)
    named_params = list(module.named_parameters())
    curr_module = exec_info.curr_module
    if named_params:
        assert curr_module in exec_info.module_to_param_usage_infos, 'The current module should have already been processed by a patched `call_module`'
        exec_info.module_to_param_usage_infos[exec_info.curr_module].append(_ParamUsageInfo(module, named_params))
    prev_curr_module = curr_module
    exec_info.curr_module = module
    exec_info.module_to_param_usage_infos[module] = []
    output = call_module(module, forward, args, kwargs)
    exec_info.curr_module = prev_curr_module
    return output