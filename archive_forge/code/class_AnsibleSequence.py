from __future__ import (absolute_import, division, print_function)
import sys as _sys
from collections.abc import Sequence
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
class AnsibleSequence(AnsibleBaseYAMLObject, list):
    """ sub class for lists """
    pass