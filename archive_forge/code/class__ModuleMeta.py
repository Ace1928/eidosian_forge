from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
class _ModuleMeta:
    """Meta information about a module.

    This class is used to represent the module information in a more structured way.
    It parses raw module information from a single item from
    `node.meta["nn_module_stack"].items()`.

    See the uses of `from_raw_meta`, `from_fx_tracer_produced_raw_meta`, and
    `from_dynamo_produced_raw_meta` for how to create an instance.

    Attributes:
        _module_class: The class of the module. E.g. `torch.nn.module.sparse.Embedding`.
        _module_name: The name of the module. E.g. `L__self___h_1_mlp_c_proj`.
        _raw_meta: The raw meta '(module_name, node.meta["nn_module_stack"][module_name])'.
    """
    _module_class: Final[Optional[type]]
    _module_name: Final[str]
    _raw_meta: Final[Tuple[Any, Any]]

    @_beartype.beartype
    def __init__(self, module_name: str, module_class: Optional[type], raw_meta: Tuple[Any, Any]):
        self._module_name = module_name
        self._module_class = module_class
        self._raw_meta = raw_meta

    @property
    def module_display_name(self) -> str:
        """The display name of the module.

        E.g. `h_1_mlp_c_proj`.
        """
        name = self.module_name
        if name.startswith('L__self___'):
            name = name[len('L__self___'):]
        return name

    @property
    def qualified_module_class_name(self) -> str:
        """Qualified name of the module class.

        E.g. `torch_nn_module_sparse_Embedding`.
        """
        if self._module_class is None:
            return ''
        return (self._module_class.__module__ + '_' + self._module_class.__name__).replace('.', '_')

    @property
    def module_class_name(self) -> str:
        """Name of the module class.

        E.g. `Embedding`.
        """
        if self._module_class is None:
            return ''
        return self._module_class.__name__

    @property
    def module_name(self) -> str:
        """Name of the module.

        E.g. `L__self___h_1_mlp_c_proj`.
        """
        return self._module_name

    @property
    def raw_meta(self) -> Tuple[Any, Any]:
        """Returns the raw module meta data.

        I.e. (module_name, node.meta['nn_module_stack'][module_name]).
        """
        return self._raw_meta

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _ModuleMeta):
            return False
        return self._module_name == __value._module_name and self._module_class == __value._module_class

    def __hash__(self) -> int:
        return hash((self._module_name, self._module_class))

    def __repr__(self) -> str:
        return f'ModuleMeta(name={self._module_name}, class={self._module_class})'

    @classmethod
    def create_root(cls) -> _ModuleMeta:
        """Create an empty module meta representing root module."""
        return _ModuleMeta('', None, ('', None))

    @classmethod
    def from_fx_tracer_produced_raw_meta(cls, raw_meta: _FX_TRACER_NN_MODULE_META_TYPE) -> _ModuleMeta:
        """Create a module meta from raw meta produced by FX symbolic tracer."""
        module_name, module_class = raw_meta
        return _ModuleMeta(module_name, module_class, raw_meta)

    @classmethod
    def from_dynamo_produced_raw_meta(cls, raw_meta: _DYNAMO_NN_MODULE_META_TYPE) -> _ModuleMeta:
        """Create a module meta from raw meta produced by FX dynamo tracer."""
        module_name, (qualified_name, module_class) = raw_meta
        return _ModuleMeta(module_name, module_class, raw_meta)

    @classmethod
    def from_raw_meta(cls, raw_meta: Union[_FX_TRACER_NN_MODULE_META_TYPE, _DYNAMO_NN_MODULE_META_TYPE]) -> _ModuleMeta:
        if isinstance(raw_meta, tuple) and len(raw_meta) == 2 and isinstance(raw_meta[1], type):
            return _ModuleMeta.from_fx_tracer_produced_raw_meta(raw_meta)
        if isinstance(raw_meta, tuple) and len(raw_meta) == 2 and isinstance(raw_meta[1], tuple):
            return _ModuleMeta.from_dynamo_produced_raw_meta(raw_meta)
        raise TypeError(f"Unknown type of raw meta item from node.meta['nn_module_stack'].items(): {type(raw_meta)}")