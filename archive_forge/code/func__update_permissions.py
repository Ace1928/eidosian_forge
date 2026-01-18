from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _update_permissions(module, keystore_path):
    """ Updates keystore file attributes as necessary """
    file_args = module.load_file_common_arguments(module.params, path=keystore_path)
    return module.set_fs_attributes_if_different(file_args, False)