from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLInputObjectField(object):
    __slots__ = ('type', 'default_value', 'description', 'out_name')

    def __init__(self, type, default_value=None, description=None, out_name=None):
        self.type = type
        self.default_value = default_value
        self.description = description
        self.out_name = out_name

    def __eq__(self, other):
        return self is other or (isinstance(other, GraphQLInputObjectField) and self.type == other.type and (self.description == other.description) and (self.out_name == other.out_name))