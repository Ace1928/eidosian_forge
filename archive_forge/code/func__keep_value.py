import json
import os
import os.path
import sys
import tempfile
import ansible.module_utils.basic
from .exceptions import (
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
def _keep_value(v, argument_specs, key, subkey=None):
    if v is None:
        return False
    if key not in argument_specs:
        return
    if not subkey:
        return v != argument_specs[key].get('default')
    elif subkey not in argument_specs[key]:
        return True
    elif isinstance(argument_specs[key][subkey], dict):
        return v != argument_specs[key][subkey].get('default')
    else:
        return True