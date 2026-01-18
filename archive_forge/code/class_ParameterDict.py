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
class ParameterDict(Module):
    """Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it
    contains are properly registered, and will be visible by all Module methods.
    Other objects are treated as would be done by a regular Python dictionary

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
    :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
    will preserve their ordering.

    Note that the constructor, assigning an element of the dictionary and the
    :meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
    :class:`~torch.nn.Parameter`.

    Args:
        values (iterable, optional): a mapping (dictionary) of
            (string : Any) or an iterable of key-value pairs
            of type (string, Any)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Any=None) -> None:
        super().__init__()
        self._keys: Dict[str, None] = {}
        if parameters is not None:
            self.update(parameters)

    def _key_to_attr(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(f"Index given to ParameterDict cannot be used as a key as it is not a string (type is '{type(key).__name__}'). Open an issue on github if you need non-string keys.")
        else:
            return key

    def __getitem__(self, key: str) -> Any:
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __setitem__(self, key: str, value: Any) -> None:
        self._keys[key] = None
        attr = self._key_to_attr(key)
        if isinstance(value, torch.Tensor) and (not isinstance(value, Parameter)):
            value = Parameter(value)
        setattr(self, attr, value)

    def __delitem__(self, key: str) -> None:
        del self._keys[key]
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __reversed__(self) -> Iterator[str]:
        return reversed(list(self._keys))

    def copy(self) -> 'ParameterDict':
        """Return a copy of this :class:`~torch.nn.ParameterDict` instance."""
        return ParameterDict(OrderedDict(((k, self[k]) for k in self._keys)))

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def setdefault(self, key: str, default: Optional[Any]=None) -> Any:
        """Set the default for a key in the Parameterdict.

        If key is in the ParameterDict, return its value.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.

        Args:
            key (str): key to set default for
            default (Any): the parameter set to the key
        """
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self) -> None:
        """Remove all items from the ParameterDict."""
        for k in self._keys.copy():
            del self[k]

    def pop(self, key: str) -> Any:
        """Remove key from the ParameterDict and return its parameter.

        Args:
            key (str): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def popitem(self) -> Tuple[str, Any]:
        """Remove and return the last inserted `(key, parameter)` pair from the ParameterDict."""
        k, _ = self._keys.popitem()
        self._keys[k] = None
        val = self[k]
        del self[k]
        return (k, val)

    def get(self, key: str, default: Optional[Any]=None) -> Any:
        """Return the parameter associated with key if present. Otherwise return default if provided, None if not.

        Args:
            key (str): key to get from the ParameterDict
            default (Parameter, optional): value to return if key not present
        """
        return self[key] if key in self else default

    def fromkeys(self, keys: Iterable[str], default: Optional[Any]=None) -> 'ParameterDict':
        """Return a new ParameterDict with the keys provided.

        Args:
            keys (iterable, string): keys to make the new ParameterDict from
            default (Parameter, optional): value to set for all keys
        """
        return ParameterDict(((k, default) for k in keys))

    def keys(self) -> Iterable[str]:
        """Return an iterable of the ParameterDict keys."""
        return self._keys.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return an iterable of the ParameterDict key/value pairs."""
        return ((k, self[k]) for k in self._keys)

    def values(self) -> Iterable[Any]:
        """Return an iterable of the ParameterDict values."""
        return (self[k] for k in self._keys)

    def update(self, parameters: Union[Mapping[str, Any], 'ParameterDict']) -> None:
        """Update the :class:`~torch.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError('ParametersDict.update should be called with an iterable of key/value pairs, but got ' + type(parameters).__name__)
        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError('ParameterDict update sequence element #' + str(j) + ' should be Iterable; is' + type(p).__name__)
                if not len(p) == 2:
                    raise ValueError('ParameterDict update sequence element #' + str(j) + ' has length ' + str(len(p)) + '; 2 is required')
                self[p[0]] = p[1]

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self.items():
            if isinstance(p, torch.Tensor):
                size_str = 'x'.join((str(size) for size in p.size()))
                if p.device.type in ['cuda', torch._C._get_privateuse1_backend_name()]:
                    device_str = f' ({p.device})'
                else:
                    device_str = ''
                parastr = '{} containing: [{} of size {}{}]'.format('Parameter' if isinstance(p, Parameter) else 'Tensor', torch.typename(p), size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
            else:
                child_lines.append('  (' + str(k) + '): Object of type: ' + type(p).__name__)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('ParameterDict should not be called.')

    def __or__(self, other: 'ParameterDict') -> 'ParameterDict':
        copy = self.copy()
        copy.update(other)
        return copy

    def __ror__(self, other: 'ParameterDict') -> 'ParameterDict':
        copy = other.copy()
        copy.update(self)
        return copy

    def __ior__(self, other: 'ParameterDict') -> Self:
        self.update(other)
        return self