import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
class ActivationWrapper(torch.nn.Module):
    """
    Base class for Activation Checkpoint and Activation Offload.

    Not meant to be instantiated directly.
    """

    def __init__(self, mod):
        super().__init__()
        self._checkpoint_wrapped_module = mod
        self._register_state_dict_hook(self._post_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook, with_module=True)

    def forward(self, *args, **kwargs):
        raise ValueError('Subclasses should implement forward().')

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._checkpoint_wrapped_module.__getitem__(key)

    def named_parameters(self, *args, **kwargs) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Override :meth:`named_parameters()` to intercept parameter names.

        remove all occurrences of ``_CHECKPOINT_PREFIX``.
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield (param_name.replace(_CHECKPOINT_PREFIX, ''), param)

    @staticmethod
    def _post_state_dict_hook(module: nn.Module, state_dict: Dict[str, Any], prefix: str, *args: Any) -> Dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.

        For ``checkpoint_wrapper``, it will strip checkpoint-wrapped module prefix,
        so that this module can be loaded into non-checkpointed modules.
        It would still be able to be loaded into checkpoint-wrapped modules as this class,
        adds the prefix back before loading the state_dict.
        """
        _replace_by_prefix(state_dict, f'{prefix}{_CHECKPOINT_PREFIX}', prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(module: nn.Module, state_dict: Dict[str, Any], prefix: str, *args: Any) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.

        For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f'{_CHECKPOINT_PREFIX}')