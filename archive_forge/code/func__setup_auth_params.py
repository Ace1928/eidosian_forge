import os
import urllib.parse
import fixtures
import openstack.config
import testtools
def _setup_auth_params(self):
    self.creds = self._credentials().get_auth_args()
    parsed_url = urllib.parse.urlparse(self.creds['auth_url'])
    auth_url = self.creds['auth_url']
    if not parsed_url.path or parsed_url.path == '/':
        auth_url = urllib.parse.urljoin(self.creds['auth_url'], 'v3')
    if parsed_url.path == '/identity':
        auth_url = '%s/v3' % auth_url
    self.conf['auth_opts']['backend'] = 'keystone'
    options = {'os_username': self.creds['username'], 'os_user_domain_id': self.creds['user_domain_id'], 'os_password': self.creds['password'], 'os_project_name': self.creds['project_name'], 'os_project_id': '', 'os_project_domain_id': self.creds['project_domain_id'], 'os_auth_url': auth_url}
    self.conf['auth_opts'].setdefault('options', {}).update(options)