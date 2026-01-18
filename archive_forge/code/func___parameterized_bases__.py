import sys
import types
import typing
from typing import (
from weakref import WeakKeyDictionary, WeakValueDictionary
from typing_extensions import Annotated, Literal as ExtLiteral
from .class_validators import gather_all_validators
from .fields import DeferredType
from .main import BaseModel, create_model
from .types import JsonWrapper
from .typing import display_as_type, get_all_type_hints, get_args, get_origin, typing_base
from .utils import all_identical, lenient_issubclass
@classmethod
def __parameterized_bases__(cls, typevars_map: Parametrization) -> Iterator[Type[Any]]:
    """
        Returns unbound bases of cls parameterised to given type variables

        :param typevars_map: Dictionary of type applications for binding subclasses.
            Given a generic class `Model` with 2 type variables [S, T]
            and a concrete model `Model[str, int]`,
            the value `{S: str, T: int}` would be passed to `typevars_map`.
        :return: an iterator of generic sub classes, parameterised by `typevars_map`
            and other assigned parameters of `cls`

        e.g.:
        ```
        class A(GenericModel, Generic[T]):
            ...

        class B(A[V], Generic[V]):
            ...

        assert A[int] in B.__parameterized_bases__({V: int})
        ```
        """

    def build_base_model(base_model: Type[GenericModel], mapped_types: Parametrization) -> Iterator[Type[GenericModel]]:
        base_parameters = tuple((mapped_types[param] for param in base_model.__parameters__))
        parameterized_base = base_model.__class_getitem__(base_parameters)
        if parameterized_base is base_model or parameterized_base is cls:
            return
        yield parameterized_base
    for base_model in cls.__bases__:
        if not issubclass(base_model, GenericModel):
            continue
        elif not getattr(base_model, '__parameters__', None):
            continue
        elif cls in _assigned_parameters:
            if base_model in _assigned_parameters:
                continue
            else:
                mapped_types: Parametrization = {key: typevars_map.get(value, value) for key, value in _assigned_parameters[cls].items()}
                yield from build_base_model(base_model, mapped_types)
        else:
            yield from build_base_model(base_model, typevars_map)