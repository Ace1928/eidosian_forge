from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _get_keystore_type_keytool_parameters(keystore_type):
    """ Check that custom keystore is presented in parameters """
    if keystore_type:
        return ['-storetype', keystore_type]
    return []