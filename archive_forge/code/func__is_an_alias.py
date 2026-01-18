import json
import os
import os.path
import sys
import tempfile
import ansible.module_utils.basic
from .exceptions import (
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
def _is_an_alias(k):
    aliases = argument_specs[k].get('aliases')
    return aliases and k in aliases