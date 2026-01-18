from urllib import parse
from manilaclient import api_versions
from manilaclient.common import httpclient
from manilaclient.tests.unit import fakes
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
def _cs_request_with_retries(self, url, method, **kwargs):
    if method in ['GET', 'DELETE']:
        assert 'body' not in kwargs
    elif method == 'PUT':
        assert 'body' in kwargs
    args = parse.parse_qsl(parse.urlparse(url)[4])
    kwargs.update(args)
    munged_url = url.rsplit('?', 1)[0]
    munged_url = munged_url.strip('/').replace('/', '_').replace('.', '_')
    munged_url = munged_url.replace('-', '_')
    callback = '%s_%s' % (method.lower(), munged_url)
    if not hasattr(self, callback):
        raise AssertionError('Called unknown API method: %s %s, expected fakes method name: %s' % (method, url, callback))
    self.callstack.append((method, url, kwargs.get('body', None)))
    status, headers, body = getattr(self, callback)(**kwargs)
    r = utils.TestResponse({'status_code': status, 'text': body, 'headers': headers})
    return (r, body)