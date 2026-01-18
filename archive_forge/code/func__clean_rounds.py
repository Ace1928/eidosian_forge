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
def _clean_rounds(self, rounds):
    algo_data = self.algorithms.get(self.algorithm)
    if rounds:
        return rounds
    elif algo_data and algo_data.implicit_rounds:
        return algo_data.implicit_rounds
    else:
        return None