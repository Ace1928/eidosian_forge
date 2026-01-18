import re
from libcloud.test import MockHttp
class BaseOvhMockHttp(MockHttp):

    def _get_method_name(self, type, use_param, qs, path):
        if type:
            meth_name = '_json{}_{}_{}'.format(FORMAT_URL.sub('_', path), 'get', type)
            return meth_name
        return '_json'

    def _json(self, method, url, body, headers):
        meth_name = '_json{}_{}'.format(FORMAT_URL.sub('_', url), method.lower())
        return getattr(self, meth_name)(method, url, body, headers)