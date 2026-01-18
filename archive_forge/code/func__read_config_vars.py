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
def _read_config_vars(self, name, **kwargs):
    """ Read configuration from variables passed to the module. """
    config = {}
    entrust_api_specification_path = kwargs.get('entrust_api_specification_path')
    if not entrust_api_specification_path or (not entrust_api_specification_path.startswith('http') and (not os.path.isfile(entrust_api_specification_path))):
        raise SessionConfigurationException(to_native("Parameter provided for entrust_api_specification_path of value '{0}' was not a valid file path or HTTPS address.".format(entrust_api_specification_path)))
    for required_file in ['entrust_api_cert', 'entrust_api_cert_key']:
        file_path = kwargs.get(required_file)
        if not file_path or not os.path.isfile(file_path):
            raise SessionConfigurationException(to_native("Parameter provided for {0} of value '{1}' was not a valid file path.".format(required_file, file_path)))
    for required_var in ['entrust_api_user', 'entrust_api_key']:
        if not kwargs.get(required_var):
            raise SessionConfigurationException(to_native('Parameter provided for {0} was missing.'.format(required_var)))
    config['entrust_api_cert'] = kwargs.get('entrust_api_cert')
    config['entrust_api_cert_key'] = kwargs.get('entrust_api_cert_key')
    config['entrust_api_specification_path'] = kwargs.get('entrust_api_specification_path')
    config['entrust_api_user'] = kwargs.get('entrust_api_user')
    config['entrust_api_key'] = kwargs.get('entrust_api_key')
    return config