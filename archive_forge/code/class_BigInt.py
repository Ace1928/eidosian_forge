from typing import Any
from graphql import Undefined
from graphql.language.ast import (
from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType
class BigInt(Scalar):
    """
    The `BigInt` scalar type represents non-fractional whole numeric values.
    `BigInt` is not constrained to 32-bit like the `Int` type and thus is a less
    compatible type.
    """

    @staticmethod
    def coerce_int(value):
        try:
            num = int(value)
        except ValueError:
            try:
                num = int(float(value))
            except ValueError:
                return Undefined
        return num
    serialize = coerce_int
    parse_value = coerce_int

    @staticmethod
    def parse_literal(ast, _variables=None):
        if isinstance(ast, IntValueNode):
            return int(ast.value)
        return Undefined