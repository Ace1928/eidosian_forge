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
def _list_deprecations(argument_spec, parameters, prefix=''):
    """Return a list of deprecations

    :arg argument_spec: An argument spec dictionary
    :arg parameters: Dictionary of parameters

    :returns: List of dictionaries containing a message and version in which
        the deprecated parameter will be removed, or an empty list.

    :Example return:

    .. code-block:: python

        [
            {
                'msg': "Param 'deptest' is deprecated. See the module docs for more information",
                'version': '2.9'
            }
        ]
    """
    deprecations = []
    for arg_name, arg_opts in argument_spec.items():
        if arg_name in parameters:
            if prefix:
                sub_prefix = '%s["%s"]' % (prefix, arg_name)
            else:
                sub_prefix = arg_name
            if arg_opts.get('removed_at_date') is not None:
                deprecations.append({'msg': "Param '%s' is deprecated. See the module docs for more information" % sub_prefix, 'date': arg_opts.get('removed_at_date'), 'collection_name': arg_opts.get('removed_from_collection')})
            elif arg_opts.get('removed_in_version') is not None:
                deprecations.append({'msg': "Param '%s' is deprecated. See the module docs for more information" % sub_prefix, 'version': arg_opts.get('removed_in_version'), 'collection_name': arg_opts.get('removed_from_collection')})
            sub_argument_spec = arg_opts.get('options')
            if sub_argument_spec is not None:
                sub_arguments = parameters[arg_name]
                if isinstance(sub_arguments, Mapping):
                    sub_arguments = [sub_arguments]
                if isinstance(sub_arguments, list):
                    for sub_params in sub_arguments:
                        if isinstance(sub_params, Mapping):
                            deprecations.extend(_list_deprecations(sub_argument_spec, sub_params, prefix=sub_prefix))
    return deprecations