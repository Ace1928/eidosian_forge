import logging
import warnings
from keystoneauth1 import identity
from keystoneauth1 import session as k_session
from monascaclient.osc import migration
from monascaclient import version
def _handle_deprecated_kwargs(kwargs):
    depr_map = {'tenant_name': ('project_name', lambda x: x), 'insecure': ('verify', lambda x: not x)}
    for key, new_key_transform in depr_map.items():
        val = kwargs.get(key, _NO_VALUE_MARKER)
        if val != _NO_VALUE_MARKER:
            new_key = new_key_transform[0]
            new_handler = new_key_transform[1]
            warnings.warn('Usage of {old_key} has been deprecated in favour of {new_key}. monascaclient will place value of {old_key} under {new_key}'.format(old_key=key, new_key=new_key), DeprecationWarning)
            kwargs[new_key] = new_handler(val)
            del kwargs[key]