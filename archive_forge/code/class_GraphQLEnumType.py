from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLEnumType(GraphQLType):
    """Enum Type Definition

    Some leaf values of requests and input values are Enums. GraphQL serializes Enum values as strings,
    however internally Enums can be represented by any kind of type, often integers.

    Example:

        RGBType = GraphQLEnumType(
            name='RGB',
            values=OrderedDict([
                ('RED', GraphQLEnumValue(0)),
                ('GREEN', GraphQLEnumValue(1)),
                ('BLUE', GraphQLEnumValue(2))
            ])
        )

    Note: If a value is not provided in a definition, the name of the enum value will be used as it's internal value.
    """

    def __init__(self, name, values, description=None):
        assert name, 'Type must provide name.'
        assert_valid_name(name)
        self.name = name
        self.description = description
        self.values = define_enum_values(self, values)

    def serialize(self, value):
        if isinstance(value, Hashable):
            enum_value = self._value_lookup.get(value)
            if enum_value:
                return enum_value.name
        return None

    def parse_value(self, value):
        if isinstance(value, Hashable):
            enum_value = self._name_lookup.get(value)
            if enum_value:
                return enum_value.value
        return None

    def parse_literal(self, value_ast):
        if isinstance(value_ast, ast.EnumValue):
            enum_value = self._name_lookup.get(value_ast.value)
            if enum_value:
                return enum_value.value

    @cached_property
    def _value_lookup(self):
        return {value.value: value for value in self.values}

    @cached_property
    def _name_lookup(self):
        return {value.name: value for value in self.values}