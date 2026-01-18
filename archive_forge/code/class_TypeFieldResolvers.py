from collections import OrderedDict, namedtuple
from ..language.printer import print_ast
from ..utils.ast_from_value import ast_from_value
from .definition import (GraphQLArgument, GraphQLEnumType, GraphQLEnumValue,
from .directives import DirectiveLocation
from .scalars import GraphQLBoolean, GraphQLString
class TypeFieldResolvers(object):
    _kinds = ((GraphQLScalarType, TypeKind.SCALAR), (GraphQLObjectType, TypeKind.OBJECT), (GraphQLInterfaceType, TypeKind.INTERFACE), (GraphQLUnionType, TypeKind.UNION), (GraphQLEnumType, TypeKind.ENUM), (GraphQLInputObjectType, TypeKind.INPUT_OBJECT), (GraphQLList, TypeKind.LIST), (GraphQLNonNull, TypeKind.NON_NULL))

    @classmethod
    def kind(cls, type, *_):
        for klass, kind in cls._kinds:
            if isinstance(type, klass):
                return kind
        raise Exception('Unknown kind of type: {}'.format(type))

    @staticmethod
    def fields(type, args, *_):
        if isinstance(type, (GraphQLObjectType, GraphQLInterfaceType)):
            fields = []
            include_deprecated = args.get('includeDeprecated')
            for field_name, field in type.fields.items():
                if field.deprecation_reason and (not include_deprecated):
                    continue
                fields.append(Field(name=field_name, description=field.description, type=field.type, args=field.args, deprecation_reason=field.deprecation_reason))
            return fields
        return None

    @staticmethod
    def interfaces(type, *_):
        if isinstance(type, GraphQLObjectType):
            return type.interfaces

    @staticmethod
    def possible_types(type, args, context, info):
        if isinstance(type, (GraphQLInterfaceType, GraphQLUnionType)):
            return info.schema.get_possible_types(type)

    @staticmethod
    def enum_values(type, args, *_):
        if isinstance(type, GraphQLEnumType):
            values = type.values
            if not args.get('includeDeprecated'):
                values = [v for v in values if not v.deprecation_reason]
            return values

    @staticmethod
    def input_fields(type, *_):
        if isinstance(type, GraphQLInputObjectType):
            return input_fields_to_list(type.fields)