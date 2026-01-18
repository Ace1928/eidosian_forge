from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def _fail_with_exc_info(self, arg0, exc):
    msg = arg0
    msg += '  More info: %s' % repr(exc)
    self.module.fail_json(msg=msg)