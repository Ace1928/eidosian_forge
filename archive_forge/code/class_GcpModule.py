from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text
class GcpModule(AnsibleModule):

    def __init__(self, *args, **kwargs):
        arg_spec = kwargs.get('argument_spec', {})
        kwargs['argument_spec'] = self._merge_dictionaries(arg_spec, dict(project=dict(required=False, type='str', fallback=(env_fallback, ['GCP_PROJECT'])), auth_kind=dict(required=True, fallback=(env_fallback, ['GCP_AUTH_KIND']), choices=['machineaccount', 'serviceaccount', 'accesstoken', 'application'], type='str'), service_account_email=dict(required=False, fallback=(env_fallback, ['GCP_SERVICE_ACCOUNT_EMAIL']), type='str'), service_account_file=dict(required=False, fallback=(env_fallback, ['GCP_SERVICE_ACCOUNT_FILE']), type='path'), service_account_contents=dict(required=False, fallback=(env_fallback, ['GCP_SERVICE_ACCOUNT_CONTENTS']), no_log=True, type='jsonarg'), access_token=dict(required=False, fallback=(env_fallback, ['GCP_ACCESS_TOKEN']), no_log=True, type='str'), scopes=dict(required=False, fallback=(env_fallback, ['GCP_SCOPES']), type='list', elements='str'), env_type=dict(required=False, fallback=(env_fallback, ['GCP_ENV_TYPE']), type='str')))
        mutual = kwargs.get('mutually_exclusive', [])
        kwargs['mutually_exclusive'] = mutual.append(['service_account_email', 'service_account_file', 'service_account_contents'])
        AnsibleModule.__init__(self, *args, **kwargs)

    def raise_for_status(self, response):
        try:
            response.raise_for_status()
        except getattr(requests.exceptions, 'RequestException') as inst:
            self.fail_json(msg='GCP returned error: %s' % response.json(), request={'url': response.request.url, 'body': response.request.body, 'method': response.request.method})

    def _merge_dictionaries(self, a, b):
        new = a.copy()
        new.update(b)
        return new