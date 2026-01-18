from __future__ import absolute_import, division, print_function
import datetime
import os
from collections import deque
from itertools import chain
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.warnings import warn
from ansible.module_utils.errors import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.six import (
from ansible.module_utils.common.validation import (
def _sanitize_keys_conditions(value, no_log_strings, ignore_keys, deferred_removals):
    """ Helper method to :func:`sanitize_keys` to build ``deferred_removals`` and avoid deep recursion. """
    if isinstance(value, (text_type, binary_type)):
        return value
    if isinstance(value, Sequence):
        if isinstance(value, MutableSequence):
            new_value = type(value)()
        else:
            new_value = []
        deferred_removals.append((value, new_value))
        return new_value
    if isinstance(value, Set):
        if isinstance(value, MutableSet):
            new_value = type(value)()
        else:
            new_value = set()
        deferred_removals.append((value, new_value))
        return new_value
    if isinstance(value, Mapping):
        if isinstance(value, MutableMapping):
            new_value = type(value)()
        else:
            new_value = {}
        deferred_removals.append((value, new_value))
        return new_value
    if isinstance(value, tuple(chain(integer_types, (float, bool, NoneType)))):
        return value
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value
    raise TypeError('Value of unknown type: %s, %s' % (type(value), value))