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
def _validate_sub_spec(argument_spec, parameters, prefix='', options_context=None, errors=None, no_log_values=None, unsupported_parameters=None, supported_parameters=None, alias_deprecations=None):
    """Validate sub argument spec.

    This function is recursive.
    """
    if options_context is None:
        options_context = []
    if errors is None:
        errors = AnsibleValidationErrorMultiple()
    if no_log_values is None:
        no_log_values = set()
    if unsupported_parameters is None:
        unsupported_parameters = set()
    if supported_parameters is None:
        supported_parameters = dict()
    for param, value in argument_spec.items():
        wanted = value.get('type')
        if wanted == 'dict' or (wanted == 'list' and value.get('elements', '') == 'dict'):
            sub_spec = value.get('options')
            if value.get('apply_defaults', False):
                if sub_spec is not None:
                    if parameters.get(param) is None:
                        parameters[param] = {}
                else:
                    continue
            elif sub_spec is None or param not in parameters or parameters[param] is None:
                continue
            options_context.append(param)
            if not isinstance(parameters[param], Sequence) or isinstance(parameters[param], string_types):
                elements = [parameters[param]]
            else:
                elements = parameters[param]
            for idx, sub_parameters in enumerate(elements):
                no_log_values.update(set_fallbacks(sub_spec, sub_parameters))
                if not isinstance(sub_parameters, dict):
                    errors.append(SubParameterTypeError("value of '%s' must be of type dict or list of dicts" % param))
                    continue
                new_prefix = prefix + param
                if wanted == 'list':
                    new_prefix += '[%d]' % idx
                new_prefix += '.'
                alias_warnings = []
                alias_deprecations_sub = []
                try:
                    options_aliases = _handle_aliases(sub_spec, sub_parameters, alias_warnings, alias_deprecations_sub)
                except (TypeError, ValueError) as e:
                    options_aliases = {}
                    errors.append(AliasError(to_native(e)))
                for option, alias in alias_warnings:
                    warn('Both option %s%s and its alias %s%s are set.' % (new_prefix, option, new_prefix, alias))
                if alias_deprecations is not None:
                    for deprecation in alias_deprecations_sub:
                        alias_deprecations.append({'name': '%s%s' % (new_prefix, deprecation['name']), 'version': deprecation.get('version'), 'date': deprecation.get('date'), 'collection_name': deprecation.get('collection_name')})
                try:
                    no_log_values.update(_list_no_log_values(sub_spec, sub_parameters))
                except TypeError as te:
                    errors.append(NoLogError(to_native(te)))
                legal_inputs = _get_legal_inputs(sub_spec, sub_parameters, options_aliases)
                unsupported_parameters.update(_get_unsupported_parameters(sub_spec, sub_parameters, legal_inputs, options_context, store_supported=supported_parameters))
                try:
                    check_mutually_exclusive(value.get('mutually_exclusive'), sub_parameters, options_context)
                except TypeError as e:
                    errors.append(MutuallyExclusiveError(to_native(e)))
                no_log_values.update(_set_defaults(sub_spec, sub_parameters, False))
                try:
                    check_required_arguments(sub_spec, sub_parameters, options_context)
                except TypeError as e:
                    errors.append(RequiredError(to_native(e)))
                _validate_argument_types(sub_spec, sub_parameters, new_prefix, options_context, errors=errors)
                _validate_argument_values(sub_spec, sub_parameters, options_context, errors=errors)
                for check in _ADDITIONAL_CHECKS:
                    try:
                        check['func'](value.get(check['attr']), sub_parameters, options_context)
                    except TypeError as e:
                        errors.append(check['err'](to_native(e)))
                no_log_values.update(_set_defaults(sub_spec, sub_parameters))
                _validate_sub_spec(sub_spec, sub_parameters, new_prefix, options_context, errors, no_log_values, unsupported_parameters, supported_parameters, alias_deprecations)
            options_context.pop()