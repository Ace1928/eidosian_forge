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
def _get_legal_inputs(argument_spec, parameters, aliases=None):
    if aliases is None:
        aliases = _handle_aliases(argument_spec, parameters)
    return list(aliases.keys()) + list(argument_spec.keys())