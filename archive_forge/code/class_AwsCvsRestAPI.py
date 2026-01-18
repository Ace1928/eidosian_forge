from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
class AwsCvsRestAPI(object):

    def __init__(self, module, timeout=60):
        self.module = module
        self.api_key = self.module.params['api_key']
        self.secret_key = self.module.params['secret_key']
        self.api_url = self.module.params['api_url']
        self.verify = self.module.params['validate_certs']
        self.timeout = timeout
        self.url = 'https://' + self.api_url + '/v1/'
        self.check_required_library()

    def check_required_library(self):
        if not HAS_REQUESTS:
            self.module.fail_json(msg=missing_required_lib('requests'))

    def send_request(self, method, api, params, json=None):
        """ send http request and process reponse, including error conditions """
        url = self.url + api
        status_code = None
        content = None
        json_dict = None
        json_error = None
        error_details = None
        headers = {'Content-type': 'application/json', 'api-key': self.api_key, 'secret-key': self.secret_key, 'Cache-Control': 'no-cache'}

        def get_json(response):
            """ extract json, and error message if present """
            try:
                json = response.json()
            except ValueError:
                return (None, None)
            success_code = [200, 201, 202]
            if response.status_code not in success_code:
                error = json.get('message')
            else:
                error = None
            return (json, error)
        try:
            response = requests.request(method, url, headers=headers, timeout=self.timeout, json=json)
            status_code = response.status_code
            json_dict, json_error = get_json(response)
        except requests.exceptions.HTTPError as err:
            __, json_error = get_json(response)
            if json_error is None:
                error_details = str(err)
        except requests.exceptions.ConnectionError as err:
            error_details = str(err)
        except Exception as err:
            error_details = str(err)
        if json_error is not None:
            error_details = json_error
        return (json_dict, error_details)

    def get(self, api, params=None):
        method = 'GET'
        return self.send_request(method, api, params)

    def post(self, api, data, params=None):
        method = 'POST'
        return self.send_request(method, api, params, json=data)

    def patch(self, api, data, params=None):
        method = 'PATCH'
        return self.send_request(method, api, params, json=data)

    def put(self, api, data, params=None):
        method = 'PUT'
        return self.send_request(method, api, params, json=data)

    def delete(self, api, data, params=None):
        method = 'DELETE'
        return self.send_request(method, api, params, json=data)

    def get_state(self, jobId):
        """ Method to get the state of the job """
        method = 'GET'
        response, status_code = self.get('Jobs/%s' % jobId)
        while str(response['state']) not in 'done':
            response, status_code = self.get('Jobs/%s' % jobId)
        return 'done'