from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _get_certificate_from_url(module, executable, url, port, pem_certificate_output):
    remote_cert_pem_chain = _download_cert_url(module, executable, url, port)
    with open(pem_certificate_output, 'w') as f:
        f.write(remote_cert_pem_chain)