from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLInterfaceType(GraphQLType):
    """Interface Type Definition

    When a field can return one of a heterogeneous set of types, a Interface type is used to describe what types are possible,
    what fields are in common across all types, as well as a function to determine which type is actually used when the field is resolved.

    Example:

        EntityType = GraphQLInterfaceType(
            name='Entity',
            fields={
                'name': GraphQLField(GraphQLString),
            })
    """

    def __init__(self, name, fields=None, resolve_type=None, description=None):
        assert name, 'Type must be named.'
        assert_valid_name(name)
        self.name = name
        self.description = description
        if resolve_type is not None:
            assert callable(resolve_type), '{} must provide "resolve_type" as a function.'.format(self)
        self.resolve_type = resolve_type
        self._fields = fields

    @cached_property
    def fields(self):
        return define_field_map(self, self._fields)