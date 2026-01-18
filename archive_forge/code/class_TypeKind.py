from collections import OrderedDict, namedtuple
from ..language.printer import print_ast
from ..utils.ast_from_value import ast_from_value
from .definition import (GraphQLArgument, GraphQLEnumType, GraphQLEnumValue,
from .directives import DirectiveLocation
from .scalars import GraphQLBoolean, GraphQLString
class TypeKind(object):
    SCALAR = 'SCALAR'
    OBJECT = 'OBJECT'
    INTERFACE = 'INTERFACE'
    UNION = 'UNION'
    ENUM = 'ENUM'
    INPUT_OBJECT = 'INPUT_OBJECT'
    LIST = 'LIST'
    NON_NULL = 'NON_NULL'