from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, cast
from datetime import date, datetime
from typing_extensions import (
import pydantic
import pydantic.generics
from pydantic.fields import FieldInfo
from ._types import (
from ._utils import (
from ._compat import (
from ._constants import RAW_RESPONSE_HEADER
def _build_discriminated_union_meta(*, union: type, meta_annotations: tuple[Any, ...]) -> DiscriminatorDetails | None:
    if isinstance(union, CachedDiscriminatorType):
        return union.__discriminator__
    discriminator_field_name: str | None = None
    for annotation in meta_annotations:
        if isinstance(annotation, PropertyInfo) and annotation.discriminator is not None:
            discriminator_field_name = annotation.discriminator
            break
    if not discriminator_field_name:
        return None
    mapping: dict[str, type] = {}
    discriminator_alias: str | None = None
    for variant in get_args(union):
        variant = strip_annotated_type(variant)
        if is_basemodel_type(variant):
            if PYDANTIC_V2:
                field = _extract_field_schema_pv2(variant, discriminator_field_name)
                if not field:
                    continue
                discriminator_alias = field.get('serialization_alias')
                field_schema = field['schema']
                if field_schema['type'] == 'literal':
                    for entry in field_schema['expected']:
                        if isinstance(entry, str):
                            mapping[entry] = variant
            else:
                field_info = cast('dict[str, FieldInfo]', variant.__fields__).get(discriminator_field_name)
                if not field_info:
                    continue
                discriminator_alias = field_info.alias
                if field_info.annotation and is_literal_type(field_info.annotation):
                    for entry in get_args(field_info.annotation):
                        if isinstance(entry, str):
                            mapping[entry] = variant
    if not mapping:
        return None
    details = DiscriminatorDetails(mapping=mapping, discriminator_field=discriminator_field_name, discriminator_alias=discriminator_alias)
    cast(CachedDiscriminatorType, union).__discriminator__ = details
    return details