from __future__ import annotations as _annotations
import operator
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import partial
from types import FunctionType
from typing import Any, Callable, Generic
import typing_extensions
from pydantic_core import PydanticUndefined, SchemaSerializer
from typing_extensions import dataclass_transform, deprecated
from ..errors import PydanticUndefinedAnnotation, PydanticUserError
from ..plugin._schema_validator import create_schema_validator
from ..warnings import GenericBeforeBaseModelWarning, PydanticDeprecatedSince20
from ._config import ConfigWrapper
from ._decorators import DecoratorInfos, PydanticDescriptorProxy, get_attribute_from_bases
from ._fields import collect_model_fields, is_valid_field_name, is_valid_privateattr_name
from ._generate_schema import GenerateSchema
from ._generics import PydanticGenericMetadata, get_model_typevars_map
from ._mock_val_ser import MockValSer, set_model_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
from ._typing_extra import get_cls_types_namespace, is_annotated, is_classvar, parent_frame_namespace
from ._utils import ClassAttribute, SafeGetItemProxy
from ._validate_call import ValidateCallWrapper
@dataclass_transform(kw_only_default=True, field_specifiers=(PydanticModelField,))
class ModelMetaclass(ABCMeta):

    def __new__(mcs, cls_name: str, bases: tuple[type[Any], ...], namespace: dict[str, Any], __pydantic_generic_metadata__: PydanticGenericMetadata | None=None, __pydantic_reset_parent_namespace__: bool=True, _create_model_module: str | None=None, **kwargs: Any) -> type:
        """Metaclass for creating Pydantic models.

        Args:
            cls_name: The name of the class to be created.
            bases: The base classes of the class to be created.
            namespace: The attribute dictionary of the class to be created.
            __pydantic_generic_metadata__: Metadata for generic models.
            __pydantic_reset_parent_namespace__: Reset parent namespace.
            _create_model_module: The module of the class to be created, if created by `create_model`.
            **kwargs: Catch-all for any other keyword arguments.

        Returns:
            The new class created by the metaclass.
        """
        if bases:
            base_field_names, class_vars, base_private_attributes = mcs._collect_bases_data(bases)
            config_wrapper = ConfigWrapper.for_model(bases, namespace, kwargs)
            namespace['model_config'] = config_wrapper.config_dict
            private_attributes = inspect_namespace(namespace, config_wrapper.ignored_types, class_vars, base_field_names)
            if private_attributes:
                original_model_post_init = get_model_post_init(namespace, bases)
                if original_model_post_init is not None:

                    def wrapped_model_post_init(self: BaseModel, __context: Any) -> None:
                        """We need to both initialize private attributes and call the user-defined model_post_init
                        method.
                        """
                        init_private_attributes(self, __context)
                        original_model_post_init(self, __context)
                    namespace['model_post_init'] = wrapped_model_post_init
                else:
                    namespace['model_post_init'] = init_private_attributes
            namespace['__class_vars__'] = class_vars
            namespace['__private_attributes__'] = {**base_private_attributes, **private_attributes}
            cls: type[BaseModel] = super().__new__(mcs, cls_name, bases, namespace, **kwargs)
            from ..main import BaseModel
            mro = cls.__mro__
            if Generic in mro and mro.index(Generic) < mro.index(BaseModel):
                warnings.warn(GenericBeforeBaseModelWarning('Classes should inherit from `BaseModel` before generic classes (e.g. `typing.Generic[T]`) for pydantic generics to work properly.'), stacklevel=2)
            cls.__pydantic_custom_init__ = not getattr(cls.__init__, '__pydantic_base_init__', False)
            cls.__pydantic_post_init__ = None if cls.model_post_init is BaseModel.model_post_init else 'model_post_init'
            cls.__pydantic_decorators__ = DecoratorInfos.build(cls)
            if __pydantic_generic_metadata__:
                cls.__pydantic_generic_metadata__ = __pydantic_generic_metadata__
            else:
                parent_parameters = getattr(cls, '__pydantic_generic_metadata__', {}).get('parameters', ())
                parameters = getattr(cls, '__parameters__', None) or parent_parameters
                if parameters and parent_parameters and (not all((x in parameters for x in parent_parameters))):
                    combined_parameters = parent_parameters + tuple((x for x in parameters if x not in parent_parameters))
                    parameters_str = ', '.join([str(x) for x in combined_parameters])
                    generic_type_label = f'typing.Generic[{parameters_str}]'
                    error_message = f'All parameters must be present on typing.Generic; you should inherit from {generic_type_label}.'
                    if Generic not in bases:
                        bases_str = ', '.join([x.__name__ for x in bases] + [generic_type_label])
                        error_message += f' Note: `typing.Generic` must go last: `class {cls.__name__}({bases_str}): ...`)'
                    raise TypeError(error_message)
                cls.__pydantic_generic_metadata__ = {'origin': None, 'args': (), 'parameters': parameters}
            cls.__pydantic_complete__ = False
            for name, obj in private_attributes.items():
                obj.__set_name__(cls, name)
            if __pydantic_reset_parent_namespace__:
                cls.__pydantic_parent_namespace__ = build_lenient_weakvaluedict(parent_frame_namespace())
            parent_namespace = getattr(cls, '__pydantic_parent_namespace__', None)
            if isinstance(parent_namespace, dict):
                parent_namespace = unpack_lenient_weakvaluedict(parent_namespace)
            types_namespace = get_cls_types_namespace(cls, parent_namespace)
            set_model_fields(cls, bases, config_wrapper, types_namespace)
            if config_wrapper.frozen and '__hash__' not in namespace:
                set_default_hash_func(cls, bases)
            complete_model_class(cls, cls_name, config_wrapper, raise_errors=False, types_namespace=types_namespace, create_model_module=_create_model_module)
            cls.model_computed_fields = {k: v.info for k, v in cls.__pydantic_decorators__.computed_fields.items()}
            super(cls, cls).__pydantic_init_subclass__(**kwargs)
            return cls
        else:
            return super().__new__(mcs, cls_name, bases, namespace, **kwargs)
    if not typing.TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            """This is necessary to keep attribute access working for class attribute access."""
            private_attributes = self.__dict__.get('__private_attributes__')
            if private_attributes and item in private_attributes:
                return private_attributes[item]
            if item == '__pydantic_core_schema__':
                maybe_mock_validator = getattr(self, '__pydantic_validator__', None)
                if isinstance(maybe_mock_validator, MockValSer):
                    rebuilt_validator = maybe_mock_validator.rebuild()
                    if rebuilt_validator is not None:
                        return getattr(self, '__pydantic_core_schema__')
            raise AttributeError(item)

    @classmethod
    def __prepare__(cls, *args: Any, **kwargs: Any) -> dict[str, object]:
        return _ModelNamespaceDict()

    def __instancecheck__(self, instance: Any) -> bool:
        """Avoid calling ABC _abc_subclasscheck unless we're pretty sure.

        See #3829 and python/cpython#92810
        """
        return hasattr(instance, '__pydantic_validator__') and super().__instancecheck__(instance)

    @staticmethod
    def _collect_bases_data(bases: tuple[type[Any], ...]) -> tuple[set[str], set[str], dict[str, ModelPrivateAttr]]:
        from ..main import BaseModel
        field_names: set[str] = set()
        class_vars: set[str] = set()
        private_attributes: dict[str, ModelPrivateAttr] = {}
        for base in bases:
            if issubclass(base, BaseModel) and base is not BaseModel:
                field_names.update(getattr(base, 'model_fields', {}).keys())
                class_vars.update(base.__class_vars__)
                private_attributes.update(base.__private_attributes__)
        return (field_names, class_vars, private_attributes)

    @property
    @deprecated('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=None)
    def __fields__(self) -> dict[str, FieldInfo]:
        warnings.warn('The `__fields__` attribute is deprecated, use `model_fields` instead.', PydanticDeprecatedSince20)
        return self.model_fields

    def __dir__(self) -> list[str]:
        attributes = list(super().__dir__())
        if '__fields__' in attributes:
            attributes.remove('__fields__')
        return attributes