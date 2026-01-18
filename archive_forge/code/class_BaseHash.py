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
class BaseHash(object):
    algo = namedtuple('algo', ['crypt_id', 'salt_size', 'implicit_rounds', 'salt_exact', 'implicit_ident'])
    algorithms = {'md5_crypt': algo(crypt_id='1', salt_size=8, implicit_rounds=None, salt_exact=False, implicit_ident=None), 'bcrypt': algo(crypt_id='2b', salt_size=22, implicit_rounds=12, salt_exact=True, implicit_ident='2b'), 'sha256_crypt': algo(crypt_id='5', salt_size=16, implicit_rounds=535000, salt_exact=False, implicit_ident=None), 'sha512_crypt': algo(crypt_id='6', salt_size=16, implicit_rounds=656000, salt_exact=False, implicit_ident=None)}

    def __init__(self, algorithm):
        self.algorithm = algorithm