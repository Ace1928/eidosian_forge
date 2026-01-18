from __future__ import (absolute_import, division, print_function)
import random
import re
import string
import sys
from collections import namedtuple
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.utils.display import Display
def _clean_salt(self, salt):
    if not salt:
        return None
    elif issubclass(self.crypt_algo.wrapped if isinstance(self.crypt_algo, PrefixWrapper) else self.crypt_algo, HasRawSalt):
        ret = to_bytes(salt, encoding='ascii', errors='strict')
    else:
        ret = to_text(salt, encoding='ascii', errors='strict')
    if self.algorithm == 'bcrypt':
        ret = bcrypt64.repair_unused(ret)
    return ret