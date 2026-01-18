from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def _deepmerge(self, d, u):
    """
        Deep merges u into d.

        Credit:
          https://bit.ly/2EDOs1B (stackoverflow question 3232943)
        License:
          cc-by-sa 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
        Changes:
          using collections_compat for compatibility

        Args:
          - d (dict): dict to merge into
          - u (dict): dict to merge into d

        Returns:
          dict, with u merged into d
        """
    for k, v in iteritems(u):
        if isinstance(v, collections_compat.Mapping):
            d[k] = self._deepmerge(d.get(k, {}), v)
        else:
            d[k] = v
    return d