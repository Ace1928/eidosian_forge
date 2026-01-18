from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def _except_hook_callback(module, original_hook, type, value, traceback):
    verbosity = _get_verbosity(module)
    if type == purefusion.rest.ApiException:
        _handle_api_exception(module, value, traceback, verbosity)
    elif type == OperationException:
        _handle_operation_exception(module, value, traceback, verbosity)
    elif issubclass(type, urllib3.exceptions.HTTPError):
        _handle_http_exception(module, value, traceback, verbosity)
    original_hook(type, value, traceback)