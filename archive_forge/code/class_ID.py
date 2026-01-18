from typing import Any
from graphql import Undefined
from graphql.language.ast import (
from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType
class ID(Scalar):
    """
    The `ID` scalar type represents a unique identifier, often used to
    refetch an object or as key for a cache. The ID type appears in a JSON
    response as a String; however, it is not intended to be human-readable.
    When expected as an input type, any string (such as `"4"`) or integer
    (such as `4`) input value will be accepted as an ID.
    """
    serialize = str
    parse_value = str

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, (StringValueNode, IntValueNode)):
            return ast.value
        return Undefined