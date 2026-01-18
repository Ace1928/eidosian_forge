from __future__ import absolute_import, division, print_function
import os
import shutil
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def _get_last_release(self, current_path):
    previous_release = None
    previous_release_path = None
    if os.path.lexists(current_path):
        previous_release_path = os.path.realpath(current_path)
        previous_release = os.path.basename(previous_release_path)
    return (previous_release, previous_release_path)