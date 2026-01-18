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
def _check_argument_values(self, spec=None, param=None):
    """ ensure all arguments have the requested values, and there are no stray arguments """
    if spec is None:
        spec = self.argument_spec
    if param is None:
        param = self.params
    for k, v in spec.items():
        choices = v.get('choices', None)
        if choices is None:
            continue
        if isinstance(choices, SEQUENCETYPE) and (not isinstance(choices, (binary_type, text_type))):
            if k in param:
                if isinstance(param[k], list):
                    diff_list = ', '.join([item for item in param[k] if item not in choices])
                    if diff_list:
                        choices_str = ', '.join([to_native(c) for c in choices])
                        msg = 'value of %s must be one or more of: %s. Got no match for: %s' % (k, choices_str, diff_list)
                        if self._options_context:
                            msg += ' found in %s' % ' -> '.join(self._options_context)
                        self.fail_json(msg=msg)
                elif param[k] not in choices:
                    lowered_choices = None
                    if param[k] == 'False':
                        lowered_choices = lenient_lowercase(choices)
                        overlap = BOOLEANS_FALSE.intersection(choices)
                        if len(overlap) == 1:
                            param[k], = overlap
                    if param[k] == 'True':
                        if lowered_choices is None:
                            lowered_choices = lenient_lowercase(choices)
                        overlap = BOOLEANS_TRUE.intersection(choices)
                        if len(overlap) == 1:
                            param[k], = overlap
                    if param[k] not in choices:
                        choices_str = ', '.join([to_native(c) for c in choices])
                        msg = 'value of %s must be one of: %s, got: %s' % (k, choices_str, param[k])
                        if self._options_context:
                            msg += ' found in %s' % ' -> '.join(self._options_context)
                        self.fail_json(msg=msg)
        else:
            msg = 'internal error: choices for argument %s are not iterable: %s' % (k, choices)
            if self._options_context:
                msg += ' found in %s' % ' -> '.join(self._options_context)
            self.fail_json(msg=msg)