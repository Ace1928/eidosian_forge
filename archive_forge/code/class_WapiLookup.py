from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
class WapiLookup(WapiBase):
    """ Implements WapiBase for lookup plugins """

    def handle_exception(self, method_name, exc):
        if 'text' in exc.response:
            raise Exception(exc.response['text'])
        else:
            raise Exception(exc)