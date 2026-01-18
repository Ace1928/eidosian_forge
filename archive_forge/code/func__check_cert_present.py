from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _check_cert_present(module, executable, keystore_path, keystore_pass, alias, keystore_type):
    """ Check if certificate with alias is present in keystore
        located at keystore_path """
    test_cmd = [executable, '-list', '-keystore', keystore_path, '-alias', alias, '-rfc']
    test_cmd += _get_keystore_type_keytool_parameters(keystore_type)
    check_rc, stdout, dummy = module.run_command(test_cmd, data=keystore_pass, check_rc=False)
    if check_rc == 0:
        return (True, stdout)
    return (False, '')