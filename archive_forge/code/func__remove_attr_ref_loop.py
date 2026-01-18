from __future__ import (absolute_import, division, print_function)
import sys
from contextlib import contextmanager
from ansible.template import Templar
from ansible.vars.manager import VariableManager
from ansible.plugins.callback.default import CallbackModule as Default
from ansible.module_utils.common.text.converters import to_text
def _remove_attr_ref_loop(obj, attributes):
    _loop_var = getattr(obj, 'loop_control', None)
    _loop_var = _loop_var or 'item'
    for attr in attributes:
        if str(_loop_var) in str(_get_value(obj=obj, attr=attr)):
            attributes.remove(attr)
    return attributes