import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
class ParametrizationList(ModuleList):
    """A sequential container that holds and manages the original parameters or buffers of a parametrized :class:`torch.nn.Module`.

    It is the type of ``module.parametrizations[tensor_name]`` when ``module[tensor_name]``
    has been parametrized with :func:`register_parametrization`.

    If the first registered parametrization has a ``right_inverse`` that returns one tensor or
    does not have a ``right_inverse`` (in which case we assume that ``right_inverse`` is the identity),
    it will hold the tensor under the name ``original``.
    If it has a ``right_inverse`` that returns more than one tensor, these will be registered as
    ``original0``, ``original1``, ...

    .. warning::
        This class is used internally by :func:`register_parametrization`. It is documented
        here for completeness. It shall not be instantiated by the user.

    Args:
        modules (sequence): sequence of modules representing the parametrizations
        original (Parameter or Tensor): parameter or buffer that is parametrized
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.
    """
    original: Tensor
    unsafe: bool

    def __init__(self, modules: Sequence[Module], original: Union[Tensor, Parameter], unsafe: bool=False) -> None:
        if len(modules) == 0:
            raise ValueError('ParametrizationList requires one or more modules.')
        super().__init__(modules)
        self.unsafe = unsafe
        original_shape = original.shape
        original_dtype = original.dtype
        with torch.no_grad():
            new = original
            for module in reversed(self):
                if hasattr(module, 'right_inverse'):
                    try:
                        new = module.right_inverse(new)
                    except NotImplementedError:
                        pass
        if not isinstance(new, Tensor) and (not isinstance(new, collections.abc.Sequence)):
            raise ValueError(f"'right_inverse' must return a Tensor or a Sequence of tensors (list, tuple...). Got {type(new).__name__}")
        self.is_tensor = isinstance(new, Tensor)
        self.ntensors = 1 if self.is_tensor else len(new)
        if self.is_tensor:
            if original.dtype != new.dtype:
                raise ValueError(f'When `right_inverse` outputs one tensor, it may not change the dtype.\noriginal.dtype: {original.dtype}\nright_inverse(original).dtype: {new.dtype}')
            with torch.no_grad():
                original.set_(new)
            _register_parameter_or_buffer(self, 'original', original)
        else:
            for i, originali in enumerate(new):
                if not isinstance(originali, Tensor):
                    raise ValueError(f"'right_inverse' must return a Tensor or a Sequence of tensors (list, tuple...). Got element {i} of the sequence with type {type(originali).__name__}.")
                if isinstance(original, Parameter):
                    originali = Parameter(originali)
                originali.requires_grad_(original.requires_grad)
                _register_parameter_or_buffer(self, f'original{i}', originali)
        if not self.unsafe:
            Z = self()
            if not isinstance(Z, Tensor):
                raise ValueError(f'A parametrization must return a tensor. Got {type(Z).__name__}.')
            if Z.dtype != original_dtype:
                raise ValueError(f'Registering a parametrization may not change the dtype of the tensor, unless `unsafe` flag is enabled.\nunparametrized dtype: {original_dtype}\nparametrized dtype: {Z.dtype}')
            if Z.shape != original_shape:
                raise ValueError(f'Registering a parametrization may not change the shape of the tensor, unless `unsafe` flag is enabled.\nunparametrized shape: {original_shape}\nparametrized shape: {Z.shape}')

    def right_inverse(self, value: Tensor) -> None:
        """Call the ``right_inverse`` methods of the parametrizations in the inverse registration order.

        Then, it stores the result in ``self.original`` if ``right_inverse`` outputs one tensor
        or in ``self.original0``, ``self.original1``, ... if it outputs several.

        Args:
            value (Tensor): Value to which initialize the module
        """
        with torch.no_grad():
            for module in reversed(self):
                if hasattr(module, 'right_inverse'):
                    value = module.right_inverse(value)
                else:
                    raise RuntimeError(f'parametrization {type(module).__name__} does not implement right_inverse.')
            if self.is_tensor:
                if not isinstance(value, Tensor):
                    raise ValueError(f'`right_inverse` should return a tensor. Got {type(value).__name__}')
                if value.dtype != self.original.dtype:
                    raise ValueError(f'The tensor returned by `right_inverse` has dtype {value.dtype} while `original` has dtype {self.original.dtype}')
                self.original.set_(value)
            else:
                if not isinstance(value, collections.abc.Sequence):
                    raise ValueError(f"'right_inverse' must return a sequence of tensors. Got {type(value).__name__}.")
                if len(value) != self.ntensors:
                    raise ValueError(f"'right_inverse' must return a sequence of tensors of length {self.ntensors}. Got a sequence of length {len(value)}.")
                for i, tensor in enumerate(value):
                    original_i = getattr(self, f'original{i}')
                    if not isinstance(tensor, Tensor):
                        raise ValueError(f'`right_inverse` must return a sequence of tensors. Got element {i} of type {type(tensor).__name__}')
                    if original_i.dtype != tensor.dtype:
                        raise ValueError(f'Tensor {i} returned by `right_inverse` has dtype {tensor.dtype} while `original{i}` has dtype {original_i.dtype}')
                    original_i.set_(tensor)

    def forward(self) -> Tensor:
        if torch.jit.is_scripting():
            raise RuntimeError('Parametrization is not working with scripting.')
        if self.is_tensor:
            x = self[0](self.original)
        else:
            originals = (getattr(self, f'original{i}') for i in range(self.ntensors))
            x = self[0](*originals)
        curr_idx = 1
        while hasattr(self, str(curr_idx)):
            x = self[curr_idx](x)
            curr_idx += 1
        return x