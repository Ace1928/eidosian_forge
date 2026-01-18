from enum import Enum
from typing import Mapping
from .definition import (
from ..language import DirectiveLocation, print_ast
from ..pyutils import inspect
from .scalars import GraphQLBoolean, GraphQLString
class InputValueFieldResolvers:

    @staticmethod
    def name(item, _info):
        return item[0]

    @staticmethod
    def description(item, _info):
        return item[1].description

    @staticmethod
    def type(item, _info):
        return item[1].type

    @staticmethod
    def default_value(item, _info):
        from ..utilities import ast_from_value
        value_ast = ast_from_value(item[1].default_value, item[1].type)
        return print_ast(value_ast) if value_ast else None

    @staticmethod
    def is_deprecated(item, _info):
        return item[1].deprecation_reason is not None

    @staticmethod
    def deprecation_reason(item, _info):
        return item[1].deprecation_reason