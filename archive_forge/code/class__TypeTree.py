from __future__ import annotations
import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple, Union
from typing_extensions import get_args, get_origin
from . import _fields, _instantiators, _resolver, _typing
from .conf import _confstruct
@dataclasses.dataclass(frozen=True)
class _TypeTree:
    typ: Any
    children: Dict[str, _TypeTree]

    @staticmethod
    def make(typ: Union[Callable, _typing.TypeForm], default_instance: _fields.DefaultInstance) -> _TypeTree:
        """From an object instance, return a data structure representing the types in the object."""
        try:
            typ, _type_from_typevar, field_list = _fields.field_list_from_callable(typ, default_instance=default_instance, support_single_arg_types=False)
        except _instantiators.UnsupportedTypeAnnotationError:
            return _TypeTree(typ, {})
        return _TypeTree(typ, {field.intern_name: _TypeTree.make(field.type_or_callable, field.default) for field in field_list})

    def is_subtype_of(self, supertype: _TypeTree) -> bool:

        def _get_type_options(typ: _typing.TypeForm) -> Tuple[_typing.TypeForm, ...]:
            return get_args(typ) if get_origin(typ) is Union else (typ,)
        self_types = _get_type_options(self.typ)
        super_types = _get_type_options(supertype.typ)
        for self_type in self_types:
            self_type = _resolver.unwrap_annotated(self_type)[0]
            self_type, _ = _resolver.unwrap_newtype(self_type)
            ok = False
            for super_type in super_types:
                super_type = _resolver.unwrap_annotated(super_type)[0]
                self_type, _ = _resolver.unwrap_newtype(self_type)
                if issubclass(self_type, super_type):
                    ok = True
            if not ok:
                return False
        return True