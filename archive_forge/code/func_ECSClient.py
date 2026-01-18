from __future__ import absolute_import, division, print_function
import json
import os
import re
import traceback
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import Request
def ECSClient(entrust_api_user=None, entrust_api_key=None, entrust_api_cert=None, entrust_api_cert_key=None, entrust_api_specification_path=None):
    """Create an ECS client"""
    if not YAML_FOUND:
        raise SessionConfigurationException(missing_required_lib('PyYAML'), exception=YAML_IMP_ERR)
    if entrust_api_specification_path is None:
        entrust_api_specification_path = 'https://cloud.entrust.net/EntrustCloud/documentation/cms-api-2.1.0.yaml'
    entrust_api_user = to_text(entrust_api_user)
    entrust_api_key = to_text(entrust_api_key)
    entrust_api_cert_key = to_text(entrust_api_cert_key)
    entrust_api_specification_path = to_text(entrust_api_specification_path)
    return ECSSession('ecs', entrust_api_user=entrust_api_user, entrust_api_key=entrust_api_key, entrust_api_cert=entrust_api_cert, entrust_api_cert_key=entrust_api_cert_key, entrust_api_specification_path=entrust_api_specification_path).client()