from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def _get_catalog_payload(params, name):
    catalog_payload = {}
    repository_type = params.get('repository_type')
    if params.get('file_name') is not None:
        catalog_payload['Filename'] = params['file_name']
    if params.get('source_path') is not None:
        catalog_payload['SourcePath'] = params['source_path']
    repository_dict = {'Name': name, 'Description': params.get('catalog_description'), 'RepositoryType': repository_type, 'Source': params.get('source'), 'CheckCertificate': params.get('check_certificate')}
    if repository_type != 'DELL_ONLINE':
        repository_dict.update({'DomainName': params.get('repository_domain'), 'Username': params.get('repository_username'), 'Password': params.get('repository_password')})
    if repository_type == 'DELL_ONLINE' and (not params.get('source')):
        repository_dict['Source'] = 'downloads.dell.com'
    repository_payload = dict([(k, v) for k, v in repository_dict.items() if v is not None])
    if repository_payload:
        catalog_payload['Repository'] = repository_payload
    return catalog_payload