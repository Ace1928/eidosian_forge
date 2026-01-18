from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLScalarType(GraphQLType):
    """Scalar Type Definition

    The leaf values of any request and input values to arguments are
    Scalars (or Enums) and are defined with a name and a series of coercion
    functions used to ensure validity.

    Example:

        def coerce_odd(value):
            if value % 2 == 1:
                return value
            return None

        OddType = GraphQLScalarType(name='Odd', serialize=coerce_odd)
    """
    __slots__ = ('name', 'description', 'serialize', 'parse_value', 'parse_literal')

    def __init__(self, name, description=None, serialize=None, parse_value=None, parse_literal=None):
        assert name, 'Type must be named.'
        assert_valid_name(name)
        self.name = name
        self.description = description
        assert callable(serialize), '{} must provide "serialize" function. If this custom Scalar is also used as an input type, ensure "parse_value" and "parse_literal" functions are also provided.'.format(self)
        if parse_value is not None or parse_literal is not None:
            assert callable(parse_value) and callable(parse_literal), '{} must provide both "parse_value" and "parse_literal" functions.'.format(self)
        self.serialize = serialize
        self.parse_value = parse_value or none_func
        self.parse_literal = parse_literal or none_func

    def __str__(self):
        return self.name