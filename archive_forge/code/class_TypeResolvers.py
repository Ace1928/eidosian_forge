from enum import Enum
from typing import Mapping
from .definition import (
from ..language import DirectiveLocation, print_ast
from ..pyutils import inspect
from .scalars import GraphQLBoolean, GraphQLString
class TypeResolvers:

    @staticmethod
    def kind(type_, _info):
        if is_scalar_type(type_):
            return TypeKind.SCALAR
        if is_object_type(type_):
            return TypeKind.OBJECT
        if is_interface_type(type_):
            return TypeKind.INTERFACE
        if is_union_type(type_):
            return TypeKind.UNION
        if is_enum_type(type_):
            return TypeKind.ENUM
        if is_input_object_type(type_):
            return TypeKind.INPUT_OBJECT
        if is_list_type(type_):
            return TypeKind.LIST
        if is_non_null_type(type_):
            return TypeKind.NON_NULL
        raise TypeError(f'Unexpected type: {inspect(type_)}.')

    @staticmethod
    def name(type_, _info):
        return getattr(type_, 'name', None)

    @staticmethod
    def description(type_, _info):
        return getattr(type_, 'description', None)

    @staticmethod
    def specified_by_url(type_, _info):
        return getattr(type_, 'specified_by_url', None)

    @staticmethod
    def fields(type_, _info, includeDeprecated=False):
        if is_object_type(type_) or is_interface_type(type_):
            items = type_.fields.items()
            return list(items) if includeDeprecated else [item for item in items if item[1].deprecation_reason is None]

    @staticmethod
    def interfaces(type_, _info):
        if is_object_type(type_) or is_interface_type(type_):
            return type_.interfaces

    @staticmethod
    def possible_types(type_, info):
        if is_abstract_type(type_):
            return info.schema.get_possible_types(type_)

    @staticmethod
    def enum_values(type_, _info, includeDeprecated=False):
        if is_enum_type(type_):
            items = type_.values.items()
            return items if includeDeprecated else [item for item in items if item[1].deprecation_reason is None]

    @staticmethod
    def input_fields(type_, _info, includeDeprecated=False):
        if is_input_object_type(type_):
            items = type_.fields.items()
            return items if includeDeprecated else [item for item in items if item[1].deprecation_reason is None]

    @staticmethod
    def of_type(type_, _info):
        return getattr(type_, 'of_type', None)