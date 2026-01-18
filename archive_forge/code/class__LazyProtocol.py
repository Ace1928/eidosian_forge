import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
class _LazyProtocol(Protocol):
    """This class is used to avoid errors with mypy checks for the attributes in a mixin.

    https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
    """

    def _register_load_state_dict_pre_hook(self, hook):
        ...

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        ...

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        ...

    def _get_name(self):
        ...

    def _infer_parameters(self, module, input):
        ...

    @property
    def _parameters(self):
        ...

    @property
    def _buffers(self):
        ...

    @property
    def _non_persistent_buffers_set(self):
        ...

    @property
    def _load_hook(self):
        ...

    @property
    def _initialize_hook(self):
        ...