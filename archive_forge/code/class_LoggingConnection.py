import os
from shlex import quote as pquote
from xml.dom.minidom import parseString
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.py3 import ensure_string
from libcloud.utils.misc import lowercase_keys
from libcloud.common.base import LibcloudConnection, HttpLibResponseProxy
class LoggingConnection(LibcloudConnection):
    """
    Debug class to log all HTTP(s) requests as they could be made
    with the curl command.

    :cvar log: file-like object that logs entries are written to.
    """
    protocol = 'https'
    log = None
    http_proxy_used = False

    def _log_response(self, r):
        rv = '# -------- begin %d:%d response ----------\n' % (id(self), id(r))
        ht = ''
        v = r.version
        if r.version == 10:
            v = 'HTTP/1.0'
        if r.version == 11:
            v = 'HTTP/1.1'
        ht += '{} {} {}\r\n'.format(v, r.status, r.reason)
        body = r.read()
        for h in r.getheaders():
            ht += '{}: {}\r\n'.format(h[0].title(), h[1])
        ht += '\r\n'
        headers = lowercase_keys(dict(r.getheaders()))
        content_type = headers.get('content-type', None)
        pretty_print = os.environ.get('LIBCLOUD_DEBUG_PRETTY_PRINT_RESPONSE', False)
        if pretty_print and content_type == 'application/json':
            try:
                body = json.loads(ensure_string(body))
                body = json.dumps(body, sort_keys=True, indent=4)
            except Exception:
                pass
        elif pretty_print and content_type in ['text/xml', 'application/xml']:
            try:
                elem = parseString(body.decode('utf-8'))
                body = elem.toprettyxml()
            except Exception:
                pass
        ht += ensure_string(body)
        rv += ht
        rv += '\n# -------- end %d:%d response ----------\n' % (id(self), id(r))
        return rv

    def _log_curl(self, method, url, body, headers):
        cmd = ['curl']
        if self.http_proxy_used:
            if self.proxy_username and self.proxy_password:
                proxy_url = '{}://{}:{}@{}:{}'.format(self.proxy_scheme, self.proxy_username, self.proxy_password, self.proxy_host, self.proxy_port)
            else:
                proxy_url = '{}://{}:{}'.format(self.proxy_scheme, self.proxy_host, self.proxy_port)
            proxy_url = pquote(proxy_url)
            cmd.extend(['--proxy', proxy_url])
        cmd.extend(['-i'])
        if method.lower() == 'head':
            cmd.extend(['--head'])
        else:
            cmd.extend(['-X', pquote(method)])
        for h in headers:
            cmd.extend(['-H', pquote('{}: {}'.format(h, headers[h]))])
        cert_file = getattr(self, 'cert_file', None)
        if cert_file:
            cmd.extend(['--cert', pquote(cert_file)])
        if body is not None and len(body) > 0:
            if isinstance(body, (bytearray, bytes)):
                body = body.decode('utf-8')
            cmd.extend(['--data-binary', pquote(body)])
        cmd.extend(['--compress'])
        cmd.extend([pquote('{}{}'.format(self.host, url))])
        return ' '.join(cmd)

    def getresponse(self):
        original_response = LibcloudConnection.getresponse(self)
        if self.log is not None:
            rv = self._log_response(HttpLibResponseProxy(original_response))
            self.log.write(u(rv + '\n'))
            self.log.flush()
        return original_response

    def request(self, method, url, body=None, headers=None, **kwargs):
        headers.update({'X-LC-Request-ID': str(id(self))})
        if self.log is not None:
            pre = '# -------- begin %d request ----------\n' % id(self)
            self.log.write(u(pre + self._log_curl(method, url, body, headers) + '\n'))
            self.log.flush()
        return LibcloudConnection.request(self, method, url, body, headers)