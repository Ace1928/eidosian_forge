from keystoneauth1 import _utils as utils
from keystoneauth1.exceptions import base
class MissingAuthMethods(base.ClientException):
    message = 'Not all required auth rules were satisfied'

    def __init__(self, response):
        self.response = response
        self.receipt = response.headers.get('Openstack-Auth-Receipt')
        body = response.json()
        self.methods = body['receipt']['methods']
        self.required_auth_methods = body['required_auth_methods']
        self.expires_at = utils.parse_isotime(body['receipt']['expires_at'])
        message = '%s: %s' % (self.message, self.required_auth_methods)
        super(MissingAuthMethods, self).__init__(message)