from __future__ import absolute_import, division, print_function
import copy
import glob
import json
import os
import re
import sys
import tempfile
import random
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.locale import get_best_parsable_locale
def _cleanup_filename(s):
    filename = self.module.params['filename']
    if filename is not None:
        return filename
    return '_'.join(re.sub('[^a-zA-Z0-9]', ' ', s).split())