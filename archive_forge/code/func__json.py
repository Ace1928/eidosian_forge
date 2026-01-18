import re
from libcloud.test import MockHttp
def _json(self, method, url, body, headers):
    meth_name = '_json{}_{}'.format(FORMAT_URL.sub('_', url), method.lower())
    return getattr(self, meth_name)(method, url, body, headers)