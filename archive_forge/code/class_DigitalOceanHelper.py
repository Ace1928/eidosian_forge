from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
class DigitalOceanHelper:
    baseurl = 'https://api.digitalocean.com/v2'

    def __init__(self, module):
        self.module = module
        self.baseurl = module.params.get('baseurl', DigitalOceanHelper.baseurl)
        self.timeout = module.params.get('timeout', 30)
        self.oauth_token = module.params.get('oauth_token')
        self.headers = {'Authorization': 'Bearer {0}'.format(self.oauth_token), 'Content-type': 'application/json'}
        response = self.get('account')
        if response.status_code == 401:
            self.module.fail_json(msg='Failed to login using API token, please verify validity of API token.')

    def _url_builder(self, path):
        if path[0] == '/':
            path = path[1:]
        return '%s/%s' % (self.baseurl, path)

    def send(self, method, path, data=None):
        url = self._url_builder(path)
        data = self.module.jsonify(data)
        if method == 'DELETE':
            if data == 'null':
                data = None
        resp, info = fetch_url(self.module, url, data=data, headers=self.headers, method=method, timeout=self.timeout)
        return Response(resp, info)

    def get(self, path, data=None):
        return self.send('GET', path, data)

    def put(self, path, data=None):
        return self.send('PUT', path, data)

    def post(self, path, data=None):
        return self.send('POST', path, data)

    def delete(self, path, data=None):
        return self.send('DELETE', path, data)

    @staticmethod
    def digital_ocean_argument_spec():
        return dict(baseurl=dict(type='str', required=False, default='https://api.digitalocean.com/v2'), validate_certs=dict(type='bool', required=False, default=True), oauth_token=dict(no_log=True, fallback=(env_fallback, ['DO_API_TOKEN', 'DO_API_KEY', 'DO_OAUTH_TOKEN', 'OAUTH_TOKEN']), required=False, aliases=['api_token']), timeout=dict(type='int', default=30))

    def get_paginated_data(self, base_url=None, data_key_name=None, data_per_page=40, expected_status_code=200):
        """
        Function to get all paginated data from given URL
        Args:
            base_url: Base URL to get data from
            data_key_name: Name of data key value
            data_per_page: Number results per page (Default: 40)
            expected_status_code: Expected returned code from DigitalOcean (Default: 200)
        Returns: List of data

        """
        page = 1
        has_next = True
        ret_data = []
        status_code = None
        response = None
        while has_next or status_code != expected_status_code:
            required_url = '{0}page={1}&per_page={2}'.format(base_url, page, data_per_page)
            response = self.get(required_url)
            status_code = response.status_code
            if status_code != expected_status_code:
                break
            page += 1
            ret_data.extend(response.json[data_key_name])
            try:
                has_next = 'pages' in response.json['links'] and 'next' in response.json['links']['pages']
            except KeyError:
                has_next = False
        if status_code != expected_status_code:
            msg = 'Failed to fetch %s from %s' % (data_key_name, base_url)
            if response:
                msg += ' due to error : %s' % response.json['message']
            self.module.fail_json(msg=msg)
        return ret_data