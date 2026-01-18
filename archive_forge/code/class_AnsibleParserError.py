from __future__ import (absolute_import, division, print_function)
import re
import traceback
from collections.abc import Sequence
from ansible.errors.yaml_strings import (
from ansible.module_utils.common.text.converters import to_native, to_text
class AnsibleParserError(AnsibleError):
    """ something was detected early that is wrong about a playbook or data file """
    pass