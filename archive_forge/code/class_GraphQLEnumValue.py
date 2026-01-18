from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLEnumValue(object):
    __slots__ = ('name', 'value', 'deprecation_reason', 'description')

    def __init__(self, value=None, deprecation_reason=None, description=None, name=None):
        self.name = name
        self.value = value
        self.deprecation_reason = deprecation_reason
        self.description = description

    def __eq__(self, other):
        return self is other or (isinstance(other, GraphQLEnumValue) and self.name == other.name and (self.value == other.value) and (self.deprecation_reason == other.deprecation_reason) and (self.description == other.description))