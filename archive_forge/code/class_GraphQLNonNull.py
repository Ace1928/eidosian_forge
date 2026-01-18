from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLNonNull(GraphQLType):
    """Non-Null Modifier

    A non-null is a kind of type marker, a wrapping type which points to another type. Non-null types enforce that their values are never null
    and can ensure an error is raised if this ever occurs during a request. It is useful for fields which you can make a strong guarantee on
    non-nullability, for example usually the id field of a database row will never be null.

    Example:

        class RowType(GraphQLObjectType):
            name = 'Row'
            fields = {
                'id': GraphQLField(type=GraphQLNonNull(GraphQLString()))
            }

    Note: the enforcement of non-nullability occurs within the executor.
    """
    __slots__ = ('of_type',)

    def __init__(self, type):
        assert is_type(type) and (not isinstance(type, GraphQLNonNull)), 'Can only create NonNull of a Nullable GraphQLType but got: {}.'.format(type)
        self.of_type = type

    def __str__(self):
        return str(self.of_type) + '!'

    def is_same_type(self, other):
        return isinstance(other, GraphQLNonNull) and self.of_type.is_same_type(other.of_type)