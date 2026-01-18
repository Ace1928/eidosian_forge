from enum import Enum
from typing import Mapping
from .definition import (
from ..language import DirectiveLocation, print_ast
from ..pyutils import inspect
from .scalars import GraphQLBoolean, GraphQLString
class FieldResolvers:

    @staticmethod
    def name(item, _info):
        return item[0]

    @staticmethod
    def description(item, _info):
        return item[1].description

    @staticmethod
    def args(item, _info, includeDeprecated=False):
        items = item[1].args.items()
        return items if includeDeprecated else [item for item in items if item[1].deprecation_reason is None]

    @staticmethod
    def type(item, _info):
        return item[1].type

    @staticmethod
    def is_deprecated(item, _info):
        return item[1].deprecation_reason is not None

    @staticmethod
    def deprecation_reason(item, _info):
        return item[1].deprecation_reason