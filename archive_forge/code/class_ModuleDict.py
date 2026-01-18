import warnings
from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice
import operator
import torch
from .module import Module
from ..parameter import Parameter
from torch._jit_internal import _copy_to_script_wrapper
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from typing_extensions import Self
class ModuleDict(Module):
    """Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """
    _modules: Dict[str, Module]

    def __init__(self, modules: Optional[Mapping[str, Module]]=None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._modules.clear()

    def pop(self, key: str) -> Module:
        """Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        """Return an iterable of the ModuleDict keys."""
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Module]]:
        """Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        """Return an iterable of the ModuleDict values."""
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        """Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError('ModuleDict.update should be called with an iterable of key/value pairs, but got ' + type(modules).__name__)
        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError('ModuleDict update sequence element #' + str(j) + ' should be Iterable; is' + type(m).__name__)
                if not len(m) == 2:
                    raise ValueError('ModuleDict update sequence element #' + str(j) + ' has length ' + str(len(m)) + '; 2 is required')
                self[m[0]] = m[1]