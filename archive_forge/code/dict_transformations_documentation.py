from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six.moves.collections_abc import MutableMapping
Recursively diff two dictionaries

    Raises ``TypeError`` for incorrect argument type.

    :arg dict1: Dictionary to compare against.
    :arg dict2: Dictionary to compare with ``dict1``.
    :return: Tuple of dictionaries of differences or ``None`` if there are no differences.
    