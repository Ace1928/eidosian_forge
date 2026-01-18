from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
@dataclass(**slots_true)
class DecoratorInfos:
    """Mapping of name in the class namespace to decorator info.

    note that the name in the class namespace is the function or attribute name
    not the field name!
    """
    validators: dict[str, Decorator[ValidatorDecoratorInfo]] = field(default_factory=dict)
    field_validators: dict[str, Decorator[FieldValidatorDecoratorInfo]] = field(default_factory=dict)
    root_validators: dict[str, Decorator[RootValidatorDecoratorInfo]] = field(default_factory=dict)
    field_serializers: dict[str, Decorator[FieldSerializerDecoratorInfo]] = field(default_factory=dict)
    model_serializers: dict[str, Decorator[ModelSerializerDecoratorInfo]] = field(default_factory=dict)
    model_validators: dict[str, Decorator[ModelValidatorDecoratorInfo]] = field(default_factory=dict)
    computed_fields: dict[str, Decorator[ComputedFieldInfo]] = field(default_factory=dict)

    @staticmethod
    def build(model_dc: type[Any]) -> DecoratorInfos:
        """We want to collect all DecFunc instances that exist as
        attributes in the namespace of the class (a BaseModel or dataclass)
        that called us
        But we want to collect these in the order of the bases
        So instead of getting them all from the leaf class (the class that called us),
        we traverse the bases from root (the oldest ancestor class) to leaf
        and collect all of the instances as we go, taking care to replace
        any duplicate ones with the last one we see to mimic how function overriding
        works with inheritance.
        If we do replace any functions we put the replacement into the position
        the replaced function was in; that is, we maintain the order.
        """
        res = DecoratorInfos()
        for base in reversed(mro(model_dc)[1:]):
            existing: DecoratorInfos | None = base.__dict__.get('__pydantic_decorators__')
            if existing is None:
                existing = DecoratorInfos.build(base)
            res.validators.update({k: v.bind_to_cls(model_dc) for k, v in existing.validators.items()})
            res.field_validators.update({k: v.bind_to_cls(model_dc) for k, v in existing.field_validators.items()})
            res.root_validators.update({k: v.bind_to_cls(model_dc) for k, v in existing.root_validators.items()})
            res.field_serializers.update({k: v.bind_to_cls(model_dc) for k, v in existing.field_serializers.items()})
            res.model_serializers.update({k: v.bind_to_cls(model_dc) for k, v in existing.model_serializers.items()})
            res.model_validators.update({k: v.bind_to_cls(model_dc) for k, v in existing.model_validators.items()})
            res.computed_fields.update({k: v.bind_to_cls(model_dc) for k, v in existing.computed_fields.items()})
        to_replace: list[tuple[str, Any]] = []
        for var_name, var_value in vars(model_dc).items():
            if isinstance(var_value, PydanticDescriptorProxy):
                info = var_value.decorator_info
                if isinstance(info, ValidatorDecoratorInfo):
                    res.validators[var_name] = Decorator.build(model_dc, cls_var_name=var_name, shim=var_value.shim, info=info)
                elif isinstance(info, FieldValidatorDecoratorInfo):
                    res.field_validators[var_name] = Decorator.build(model_dc, cls_var_name=var_name, shim=var_value.shim, info=info)
                elif isinstance(info, RootValidatorDecoratorInfo):
                    res.root_validators[var_name] = Decorator.build(model_dc, cls_var_name=var_name, shim=var_value.shim, info=info)
                elif isinstance(info, FieldSerializerDecoratorInfo):
                    for field_serializer_decorator in res.field_serializers.values():
                        if field_serializer_decorator.cls_var_name == var_name:
                            continue
                        for f in info.fields:
                            if f in field_serializer_decorator.info.fields:
                                raise PydanticUserError(f'Multiple field serializer functions were defined for field {f!r}, this is not allowed.', code='multiple-field-serializers')
                    res.field_serializers[var_name] = Decorator.build(model_dc, cls_var_name=var_name, shim=var_value.shim, info=info)
                elif isinstance(info, ModelValidatorDecoratorInfo):
                    res.model_validators[var_name] = Decorator.build(model_dc, cls_var_name=var_name, shim=var_value.shim, info=info)
                elif isinstance(info, ModelSerializerDecoratorInfo):
                    res.model_serializers[var_name] = Decorator.build(model_dc, cls_var_name=var_name, shim=var_value.shim, info=info)
                else:
                    from ..fields import ComputedFieldInfo
                    isinstance(var_value, ComputedFieldInfo)
                    res.computed_fields[var_name] = Decorator.build(model_dc, cls_var_name=var_name, shim=None, info=info)
                to_replace.append((var_name, var_value.wrapped))
        if to_replace:
            setattr(model_dc, '__pydantic_decorators__', res)
            for name, value in to_replace:
                setattr(model_dc, name, value)
        return res