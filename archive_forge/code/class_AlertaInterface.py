from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils.basic import AnsibleModule
class AlertaInterface(object):

    def __init__(self, module):
        self.module = module
        self.state = module.params['state']
        self.customer = module.params['customer']
        self.match = module.params['match']
        self.alerta_url = module.params['alerta_url']
        self.headers = {'Content-Type': 'application/json'}
        if module.params.get('api_key', None):
            self.headers['Authorization'] = 'Key %s' % module.params['api_key']
        else:
            self.headers['Authorization'] = basic_auth_header(module.params['api_username'], module.params['api_password'])

    def send_request(self, url, data=None, method='GET'):
        response, info = fetch_url(self.module, url, data=data, headers=self.headers, method=method)
        status_code = info['status']
        if status_code == 401:
            self.module.fail_json(failed=True, response=info, msg="Unauthorized to request '%s' on '%s'" % (method, url))
        elif status_code == 403:
            self.module.fail_json(failed=True, response=info, msg="Permission Denied for '%s' on '%s'" % (method, url))
        elif status_code == 404:
            self.module.fail_json(failed=True, response=info, msg="Not found for request '%s' on '%s'" % (method, url))
        elif status_code in (200, 201):
            return self.module.from_json(response.read())
        self.module.fail_json(failed=True, response=info, msg='Alerta API error with HTTP %d for %s' % (status_code, url))

    def get_customers(self):
        url = '%s/api/customers' % self.alerta_url
        response = self.send_request(url)
        pages = response['pages']
        if pages > 1:
            for page in range(2, pages + 1):
                page_url = url + '?page=' + str(page)
                new_results = self.send_request(page_url)
                response.update(new_results)
        return response

    def create_customer(self):
        url = '%s/api/customer' % self.alerta_url
        payload = {'customer': self.customer, 'match': self.match}
        payload = self.module.jsonify(payload)
        response = self.send_request(url, payload, 'POST')
        return response

    def delete_customer(self, id):
        url = '%s/api/customer/%s' % (self.alerta_url, id)
        response = self.send_request(url, None, 'DELETE')
        return response

    def find_customer_id(self, customer):
        for i in customer['customers']:
            if self.customer == i['customer'] and self.match == i['match']:
                return i['id']
        return None