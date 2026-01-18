from __future__ import (absolute_import, division, print_function)
import shlex
from abc import abstractmethod
from random import choice
from string import ascii_lowercase
from gettext import dgettext
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins import AnsiblePlugin
def _check_password_error(self, b_out, msg):
    """ returns True/False if domain specific i18n version of msg is found in b_out """
    b_fail = to_bytes(dgettext(self.name, msg))
    return b_fail and b_fail in b_out