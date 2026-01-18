from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
class WapiBase(object):
    """ Base class for implementing Infoblox WAPI API """
    provider_spec = {'provider': dict(type='dict', options=NIOS_PROVIDER_SPEC)}

    def __init__(self, provider):
        self.connector = get_connector(**provider)

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            if name.startswith('_'):
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
            return partial(self._invoke_method, name)

    def _invoke_method(self, name, *args, **kwargs):
        try:
            method = getattr(self.connector, name)
            return method(*args, **kwargs)
        except InfobloxException as exc:
            if hasattr(self, 'handle_exception'):
                self.handle_exception(name, exc)
            else:
                raise