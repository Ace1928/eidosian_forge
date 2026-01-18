from __future__ import absolute_import, division, print_function
import datetime
import os
from collections import deque
from itertools import chain
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.warnings import warn
from ansible.module_utils.errors import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.six import (
from ansible.module_utils.common.validation import (
def _validate_elements(wanted_type, parameter, values, options_context=None, errors=None):
    if errors is None:
        errors = AnsibleValidationErrorMultiple()
    type_checker, wanted_element_type = _get_type_validator(wanted_type)
    validated_parameters = []
    kwargs = {}
    if wanted_element_type == 'str' and isinstance(wanted_type, string_types):
        if isinstance(parameter, string_types):
            kwargs['param'] = parameter
        elif isinstance(parameter, dict):
            kwargs['param'] = list(parameter.keys())[0]
    for value in values:
        try:
            validated_parameters.append(type_checker(value, **kwargs))
        except (TypeError, ValueError) as e:
            msg = "Elements value for option '%s'" % parameter
            if options_context:
                msg += " found in '%s'" % ' -> '.join(options_context)
            msg += ' is of type %s and we were unable to convert to %s: %s' % (type(value), wanted_element_type, to_native(e))
            errors.append(ElementError(msg))
    return validated_parameters