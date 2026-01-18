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
def _check_argument_types(self, spec=None, param=None, prefix=''):
    """ ensure all arguments have the requested type """
    if spec is None:
        spec = self.argument_spec
    if param is None:
        param = self.params
    for k, v in spec.items():
        wanted = v.get('type', None)
        if k not in param:
            continue
        value = param[k]
        if value is None:
            continue
        type_checker, wanted_name = self._get_wanted_type(wanted, k)
        kwargs = {}
        if wanted_name == 'str' and isinstance(type_checker, string_types):
            kwargs['param'] = list(param.keys())[0]
            if prefix:
                kwargs['prefix'] = prefix
        try:
            param[k] = type_checker(value, **kwargs)
            wanted_elements = v.get('elements', None)
            if wanted_elements:
                if wanted != 'list' or not isinstance(param[k], list):
                    msg = "Invalid type %s for option '%s'" % (wanted_name, param)
                    if self._options_context:
                        msg += " found in '%s'." % ' -> '.join(self._options_context)
                    msg += ", elements value check is supported only with 'list' type"
                    self.fail_json(msg=msg)
                param[k] = self._handle_elements(wanted_elements, k, param[k])
        except (TypeError, ValueError) as e:
            msg = 'argument %s is of type %s' % (k, type(value))
            if self._options_context:
                msg += " found in '%s'." % ' -> '.join(self._options_context)
            msg += ' and we were unable to convert to %s: %s' % (wanted_name, to_native(e))
            self.fail_json(msg=msg)