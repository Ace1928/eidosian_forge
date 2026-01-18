from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _download_cert_url(module, executable, url, port):
    """ Fetches the certificate from the remote URL using `keytool -printcert...`
          The PEM formatted string is returned """
    proxy_opts = build_proxy_options()
    fetch_cmd = [executable, '-printcert', '-rfc', '-sslserver'] + proxy_opts + ['%s:%d' % (url, port)]
    fetch_rc, fetch_out, fetch_err = module.run_command(fetch_cmd, check_rc=False)
    if fetch_rc != 0:
        module.fail_json(msg='Internal module failure, cannot download certificate, error: %s' % fetch_err, rc=fetch_rc, cmd=fetch_cmd)
    return fetch_out