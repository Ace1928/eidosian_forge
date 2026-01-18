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
def _get_type_validator(wanted):
    """Returns the callable used to validate a wanted type and the type name.

    :arg wanted: String or callable. If a string, get the corresponding
        validation function from DEFAULT_TYPE_VALIDATORS. If callable,
        get the name of the custom callable and return that for the type_checker.

    :returns: Tuple of callable function or None, and a string that is the name
        of the wanted type.
    """
    if not callable(wanted):
        if wanted is None:
            wanted = 'str'
        type_checker = DEFAULT_TYPE_VALIDATORS.get(wanted)
    else:
        type_checker = wanted
        wanted = getattr(wanted, '__name__', to_native(type(wanted)))
    return (type_checker, wanted)