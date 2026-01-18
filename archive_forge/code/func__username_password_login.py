from __future__ import absolute_import, division, print_function
from . import errors, http
def _username_password_login(self):
    resp = http.request('GET', '{0}/auth'.format(self.address), force_basic_auth=True, url_username=self.username, url_password=self.password, validate_certs=self.verify, ca_path=self.ca_path)
    if resp.status != 200:
        raise errors.SensuError('Authentication call returned status {0}'.format(resp.status))
    if resp.json is None:
        raise errors.SensuError('Authentication call did not return a valid JSON')
    if 'access_token' not in resp.json:
        raise errors.SensuError('Authentication call did not return access token')
    return dict(Authorization='Bearer {0}'.format(resp.json['access_token']))