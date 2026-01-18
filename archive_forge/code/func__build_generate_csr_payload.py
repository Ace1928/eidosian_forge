from __future__ import (absolute_import, division, print_function)
import json
import base64
import os
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import reset_idrac
def _build_generate_csr_payload(module, cert_type):
    payload = {}
    cert_params = module.params.get('cert_params')
    for key, value in csr_transform.items():
        if cert_params.get(key) is not None:
            payload[value] = cert_params.get(key)
    if rfish_cert_coll.get(cert_type):
        payload['CertificateCollection'] = rfish_cert_coll.get(cert_type)
    return payload