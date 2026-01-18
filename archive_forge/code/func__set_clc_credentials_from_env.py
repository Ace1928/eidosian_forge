from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _set_clc_credentials_from_env(self):
    """
        Set the CLC Credentials on the sdk by reading environment variables
        :return: none
        """
    env = os.environ
    v2_api_token = env.get('CLC_V2_API_TOKEN', False)
    v2_api_username = env.get('CLC_V2_API_USERNAME', False)
    v2_api_passwd = env.get('CLC_V2_API_PASSWD', False)
    clc_alias = env.get('CLC_ACCT_ALIAS', False)
    api_url = env.get('CLC_V2_API_URL', False)
    if api_url:
        self.clc.defaults.ENDPOINT_URL_V2 = api_url
    if v2_api_token and clc_alias:
        self.clc._LOGIN_TOKEN_V2 = v2_api_token
        self.clc._V2_ENABLED = True
        self.clc.ALIAS = clc_alias
    elif v2_api_username and v2_api_passwd:
        self.clc.v2.SetCredentials(api_username=v2_api_username, api_passwd=v2_api_passwd)
    else:
        return self.module.fail_json(msg='You must set the CLC_V2_API_USERNAME and CLC_V2_API_PASSWD environment variables')