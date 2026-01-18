from __future__ import (absolute_import, division, print_function)
import sys
from contextlib import contextmanager
from ansible.template import Templar
from ansible.vars.manager import VariableManager
from ansible.plugins.callback.default import CallbackModule as Default
from ansible.module_utils.common.text.converters import to_text
def _get_output_specification(self, loader, variables):
    _ret = {}
    _calling_method = sys._getframe(1).f_code.co_name
    _callback_type = _calling_method[3:] if _calling_method[:3] == 'v2_' else _calling_method
    _callback_options = ['msg', 'msg_color']
    for option in _callback_options:
        _option_name = '%s_%s' % (_callback_type, option)
        _option_template = variables.get(self.DIY_NS + '_' + _option_name, self.get_option(_option_name))
        _ret.update({option: self._template(loader=loader, template=_option_template, variables=variables)})
    _ret.update({'vars': variables})
    return _ret