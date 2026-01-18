import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def ExpandRelativePath(method_config, params, relative_path=None):
    """Determine the relative path for request."""
    path = relative_path or method_config.relative_path or ''
    for param in method_config.path_params:
        param_template = '{%s}' % param
        reserved_chars = ''
        reserved_template = '{+%s}' % param
        if reserved_template in path:
            reserved_chars = _RESERVED_URI_CHARS
            path = path.replace(reserved_template, param_template)
        if param_template not in path:
            raise exceptions.InvalidUserInputError('Missing path parameter %s' % param)
        try:
            value = params[param]
        except KeyError:
            raise exceptions.InvalidUserInputError('Request missing required parameter %s' % param)
        if value is None:
            raise exceptions.InvalidUserInputError('Request missing required parameter %s' % param)
        try:
            if not isinstance(value, six.string_types):
                value = str(value)
            path = path.replace(param_template, urllib_parse.quote(value.encode('utf_8'), reserved_chars))
        except TypeError as e:
            raise exceptions.InvalidUserInputError('Error setting required parameter %s to value %s: %s' % (param, value, e))
    return path