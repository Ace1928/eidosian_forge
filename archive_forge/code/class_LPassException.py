from __future__ import (absolute_import, division, print_function)
from subprocess import Popen, PIPE
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.lookup import LookupBase
class LPassException(AnsibleError):
    pass