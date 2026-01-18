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
def _get_unsupported_parameters(argument_spec, parameters, legal_inputs=None, options_context=None, store_supported=None):
    """Check keys in parameters against those provided in legal_inputs
    to ensure they contain legal values. If legal_inputs are not supplied,
    they will be generated using the argument_spec.

    :arg argument_spec: Dictionary of parameters, their type, and valid values.
    :arg parameters: Dictionary of parameters.
    :arg legal_inputs: List of valid key names property names. Overrides values
        in argument_spec.
    :arg options_context: List of parent keys for tracking the context of where
        a parameter is defined.

    :returns: Set of unsupported parameters. Empty set if no unsupported parameters
        are found.
    """
    if legal_inputs is None:
        legal_inputs = _get_legal_inputs(argument_spec, parameters)
    unsupported_parameters = set()
    for k in parameters.keys():
        if k not in legal_inputs:
            context = k
            if options_context:
                context = tuple(options_context + [k])
            unsupported_parameters.add(context)
            if store_supported is not None:
                supported_aliases = _handle_aliases(argument_spec, parameters)
                supported_params = []
                for option in legal_inputs:
                    if option in supported_aliases:
                        continue
                    supported_params.append(option)
                store_supported.update({context: (supported_params, supported_aliases)})
    return unsupported_parameters