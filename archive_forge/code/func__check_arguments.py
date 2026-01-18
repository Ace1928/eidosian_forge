from __future__ import absolute_import, division, print_function
import abc
import copy
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleFallbackNotFound, SEQUENCETYPE, remove_values
from ansible.module_utils.common._collections_compat import (
from ansible.module_utils.common.parameters import (
from ansible.module_utils.common.validation import (
from ansible.module_utils.common.text.formatters import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.action import ActionBase
def _check_arguments(self, spec=None, param=None, legal_inputs=None):
    self._syslog_facility = 'LOG_USER'
    unsupported_parameters = set()
    if spec is None:
        spec = self.argument_spec
    if param is None:
        param = self.params
    if legal_inputs is None:
        legal_inputs = self._legal_inputs
    for k in list(param.keys()):
        if k not in legal_inputs:
            unsupported_parameters.add(k)
    for k in PASS_VARS:
        param_key = '_ansible_%s' % k
        if param_key in param:
            if k in PASS_BOOLS:
                setattr(self, PASS_VARS[k][0], self.boolean(param[param_key]))
            else:
                setattr(self, PASS_VARS[k][0], param[param_key])
            if param_key in self.params:
                del self.params[param_key]
        elif not hasattr(self, PASS_VARS[k][0]):
            setattr(self, PASS_VARS[k][0], PASS_VARS[k][1])
    if unsupported_parameters:
        msg = 'Unsupported parameters for (%s) module: %s' % (self._name, ', '.join(sorted(list(unsupported_parameters))))
        if self._options_context:
            msg += ' found in %s.' % ' -> '.join(self._options_context)
        supported_parameters = list()
        for key in sorted(spec.keys()):
            if 'aliases' in spec[key] and spec[key]['aliases']:
                supported_parameters.append('%s (%s)' % (key, ', '.join(sorted(spec[key]['aliases']))))
            else:
                supported_parameters.append(key)
        msg += ' Supported parameters include: %s' % ', '.join(supported_parameters)
        self.fail_json(msg=msg)
    if self.check_mode and (not self.supports_check_mode):
        self.exit_json(skipped=True, msg='action module (%s) does not support check mode' % self._name)