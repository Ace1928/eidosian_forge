from __future__ import absolute_import, division, print_function
import os
import shutil
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def _get_file_args(self, path):
    file_args = self.file_args.copy()
    file_args['path'] = path
    return file_args