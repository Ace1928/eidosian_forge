from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
class defaultdict(_defaultdict):

    def __eq__(self, other):
        return super(defaultdict, self).__eq__(other) and isinstance(other, _defaultdict) and (self.default_factory == other.default_factory)