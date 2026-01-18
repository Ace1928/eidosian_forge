from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def _handle_api_exception(module, exception, traceback, verbosity):
    error_message, body = format_fusion_api_exception(exception, traceback)
    if verbosity > 1:
        module.fail_json(msg=error_message, call_details=body, traceback=str(traceback))
    elif verbosity > 0:
        module.fail_json(msg=error_message, call_details=body)
    else:
        module.fail_json(msg=error_message)