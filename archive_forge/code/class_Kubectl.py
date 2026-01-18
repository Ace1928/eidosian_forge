from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class Kubectl(object):

    def __init__(self, module):
        self.module = module
    '\n    Writes a kubectl config file\n    kubectl_path must be set or this will fail.\n    '

    def write_file(self):
        try:
            import yaml
        except ImportError:
            self.module.fail_json(msg='Please install the pyyaml module')
        with open(self.module.params['kubectl_path'], 'w') as f:
            f.write(yaml.dump(self._contents()))
    '\n    Returns the contents of a kubectl file\n    '

    def _contents(self):
        token = self._auth_token()
        endpoint = 'https://%s' % self.fetch['endpoint']
        context = self.module.params.get('kubectl_context')
        if not context:
            context = self.module.params['name']
        user = {'name': context, 'user': {'auth-provider': {'config': {'access-token': token, 'cmd-args': 'config config-helper --format=json', 'cmd-path': '/usr/lib64/google-cloud-sdk/bin/gcloud', 'expiry-key': '{.credential.token_expiry}', 'token-key': '{.credential.access_token}'}, 'name': 'gcp'}}}
        auth_keyword = self.fetch['masterAuth'].keys()
        if 'username' in auth_keyword and 'password' in auth_keyword:
            user['user']['auth-provider'].update({'username': str(self.fetch['masterAuth']['username']), 'password': str(self.fetch['masterAuth']['password'])})
        return {'apiVersion': 'v1', 'clusters': [{'name': context, 'cluster': {'certificate-authority-data': str(self.fetch['masterAuth']['clusterCaCertificate'])}}], 'contexts': [{'name': context, 'context': {'cluster': context, 'user': context}}], 'current-context': context, 'kind': 'Config', 'preferences': {}, 'users': [user]}
    "\n    Returns the auth token used in kubectl\n    This also sets the 'fetch' variable used in creating the kubectl\n    "

    def _auth_token(self):
        auth = GcpSession(self.module, 'auth')
        response = auth.get(self_link(self.module))
        self.fetch = response.json()
        return response.request.headers['authorization'].split(' ')[1]