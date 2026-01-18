from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
@cached_property
def _value_lookup(self):
    return {value.value: value for value in self.values}