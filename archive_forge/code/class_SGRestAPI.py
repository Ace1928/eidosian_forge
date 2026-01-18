from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
class SGRestAPI(object):

    def __init__(self, module, timeout=60):
        self.module = module
        self.auth_token = self.module.params['auth_token']
        self.api_url = self.module.params['api_url']
        self.verify = self.module.params['validate_certs']
        self.timeout = timeout
        self.check_required_library()
        self.sg_version = dict(major=-1, minor=-1, full='', valid=False)

    def check_required_library(self):
        if not HAS_REQUESTS:
            self.module.fail_json(msg=missing_required_lib('requests'))

    def send_request(self, method, api, params, json=None):
        """send http request and process reponse, including error conditions"""
        url = '%s/%s' % (self.api_url, api)
        status_code = None
        content = None
        json_dict = None
        json_error = None
        error_details = None
        headers = {'Content-type': 'application/json', 'Authorization': self.auth_token, 'Cache-Control': 'no-cache'}

        def get_json(response):
            """extract json, and error message if present"""
            try:
                json = response.json()
            except ValueError:
                return (None, None)
            success_code = [200, 201, 202, 204]
            if response.status_code not in success_code:
                error = json.get('message')
            else:
                error = None
            return (json, error)
        try:
            response = requests.request(method, url, headers=headers, timeout=self.timeout, json=json, verify=self.verify, params=params)
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

    def get_sg_product_version(self, api_root='grid'):
        method = 'GET'
        api = 'api/v3/%s/config/product-version' % api_root
        message, error = self.send_request(method, api, params={})
        if error:
            self.module.fail_json(msg=error)
        self.set_version(message)

    def set_version(self, message):
        try:
            product_version = message.get('data', 'not found').get('productVersion', 'not_found')
        except AttributeError:
            self.sg_version['valid'] = False
            return
        self.sg_version['major'], self.sg_version['minor'] = list(map(int, product_version.split('.')[0:2]))
        self.sg_version['full'] = product_version
        self.sg_version['valid'] = True

    def get_sg_version(self):
        if self.sg_version['valid']:
            return (self.sg_version['major'], self.sg_version['minor'])
        return (-1, -1)

    def meets_sg_minimum_version(self, minimum_major, minimum_minor):
        return self.get_sg_version() >= (minimum_major, minimum_minor)

    def requires_sg_version(self, module_or_option, version):
        return '%s requires StorageGRID %s or later.' % (module_or_option, version)

    def fail_if_not_sg_minimum_version(self, module_or_option, minimum_major, minimum_minor):
        version = self.get_sg_version()
        if version < (minimum_major, minimum_minor):
            msg = 'Error: ' + self.requires_sg_version(module_or_option, '%d.%d' % (minimum_major, minimum_minor))
            msg += '  Found: %s.%s.' % version
            self.module.fail_json(msg=msg)