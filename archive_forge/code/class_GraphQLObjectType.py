from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLObjectType(GraphQLType):
    """Object Type Definition

    Almost all of the GraphQL types you define will be object types.
    Object types have a name, but most importantly describe their fields.

    Example:

        AddressType = GraphQLObjectType('Address', {
            'street': GraphQLField(GraphQLString),
            'number': GraphQLField(GraphQLInt),
            'formatted': GraphQLField(GraphQLString,
                resolver=lambda obj, args, context, info: obj.number + ' ' + obj.street),
        })

    When two types need to refer to each other, or a type needs to refer to
    itself in a field, you can use a static method to supply the fields
    lazily.

    Example:

        PersonType = GraphQLObjectType('Person', lambda: {
            'name': GraphQLField(GraphQLString),
            'bestFriend': GraphQLField(PersonType)
        })
    """

    def __init__(self, name, fields, interfaces=None, is_type_of=None, description=None):
        assert name, 'Type must be named.'
        assert_valid_name(name)
        self.name = name
        self.description = description
        if is_type_of is not None:
            assert callable(is_type_of), '{} must provide "is_type_of" as a function.'.format(self)
        self.is_type_of = is_type_of
        self._fields = fields
        self._provided_interfaces = interfaces
        self._interfaces = None

    @cached_property
    def fields(self):
        return define_field_map(self, self._fields)

    @cached_property
    def interfaces(self):
        return define_interfaces(self, self._provided_interfaces)