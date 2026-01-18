from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _format_call_signature(name, args, kwargs):
    message = '%s(%%s)' % name
    formatted_args = ''
    args_string = ', '.join([repr(arg) for arg in args])

    def encode_item(item):
        if six.PY2 and isinstance(item, unicode):
            return item.encode('utf-8')
        else:
            return item
    kwargs_string = ', '.join(['%s=%r' % (encode_item(key), value) for key, value in sorted(kwargs.items())])
    if args_string:
        formatted_args = args_string
    if kwargs_string:
        if formatted_args:
            formatted_args += ', '
        formatted_args += kwargs_string
    return message % formatted_args