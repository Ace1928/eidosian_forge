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
def _check_type_str(self, value, param=None, prefix=''):
    opts = {'error': False, 'warn': False, 'ignore': True}
    allow_conversion = opts.get(C.STRING_CONVERSION_ACTION, True)
    try:
        return check_type_str(value, allow_conversion)
    except TypeError:
        common_msg = 'quote the entire value to ensure it does not change.'
        from_msg = '{0!r}'.format(value)
        to_msg = '{0!r}'.format(to_text(value))
        if param is not None:
            if prefix:
                param = '{0}{1}'.format(prefix, param)
            from_msg = '{0}: {1!r}'.format(param, value)
            to_msg = '{0}: {1!r}'.format(param, to_text(value))
        if C.STRING_CONVERSION_ACTION == 'error':
            msg = common_msg.capitalize()
            raise TypeError(to_native(msg))
        elif C.STRING_CONVERSION_ACTION == 'warn':
            msg = 'The value "{0}" (type {1.__class__.__name__}) was converted to "{2}" (type string). If this does not look like what you expect, {3}'.format(from_msg, value, to_msg, common_msg)
            self.warn(to_native(msg))
            return to_native(value, errors='surrogate_or_strict')