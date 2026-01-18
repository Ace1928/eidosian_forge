from libcloud.test import MockHttp
from libcloud.utils.py3 import xmlrpclib
class BaseGandiMockHttp(MockHttp):

    def _get_method_name(self, type, use_param, qs, path):
        return '_xmlrpc'

    def _xmlrpc(self, method, url, body, headers):
        params, methodName = xmlrpclib.loads(body)
        meth_name = '_xmlrpc__' + methodName.replace('.', '_')
        if self.type:
            meth_name = '{}_{}'.format(meth_name, self.type)
        return getattr(self, meth_name)(method, url, body, headers)