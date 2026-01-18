from __future__ import (absolute_import, division, print_function)
import os.path
from ansible import constants as C
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import MutableSequence
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.plugins.loader import connection_loader
def _host_is_ipv6_address(self, host):
    return ':' in to_text(host, errors='surrogate_or_strict')