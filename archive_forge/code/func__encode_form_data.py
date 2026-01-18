from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def _encode_form_data(payload_file):
    """Encode multipart/form-data for file upload."""
    fields = []
    f_name, f_data, f_type = payload_file.get('file')
    f_binary = f_data.read()
    req_field = RequestField(name='file', data=f_binary, filename=f_name)
    req_field.make_multipart(content_type=f_type)
    fields.append(req_field)
    data, content_type = encode_multipart_formdata(fields)
    return (data, content_type)