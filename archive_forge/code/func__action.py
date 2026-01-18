from urllib import parse
from zunclient import api_versions
from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def _action(self, id, action, method='POST', qparams=None, **kwargs):
    if qparams:
        action = '%s?%s' % (action, parse.urlencode(qparams))
    kwargs.setdefault('headers', {})
    kwargs['headers'].setdefault('Content-Length', '0')
    resp, body = self.api.json_request(method, self._path(id) + action, **kwargs)
    return (resp, body)