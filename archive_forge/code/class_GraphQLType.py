from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLType(object):
    __slots__ = ('name',)

    def __str__(self):
        return self.name

    def is_same_type(self, other):
        return self.__class__ is other.__class__ and self.name == other.name