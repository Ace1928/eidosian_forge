from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _export_public_cert_from_pkcs12(module, executable, pkcs_file, alias, password, dest):
    """ Runs keytools to extract the public cert from a PKCS12 archive and write it to a file. """
    export_cmd = [executable, '-list', '-noprompt', '-keystore', pkcs_file, '-alias', alias, '-storetype', 'pkcs12', '-rfc']
    export_rc, export_stdout, export_err = module.run_command(export_cmd, data=password, check_rc=False)
    if export_rc != 0:
        module.fail_json(msg='Internal module failure, cannot extract public certificate from PKCS12, message: %s' % export_stdout, stderr=export_err, rc=export_rc)
    with open(dest, 'w') as f:
        f.write(export_stdout)