import hashlib
import hmac
import os
import six
from ._cookiejar import SimpleCookieJar
from ._exceptions import *
from ._http import *
from ._logging import *
from ._socket import *
def _get_handshake_headers(resource, host, port, options):
    headers = ['GET %s HTTP/1.1' % resource, 'Upgrade: websocket', 'Connection: Upgrade']
    if port == 80 or port == 443:
        hostport = _pack_hostname(host)
    else:
        hostport = '%s:%d' % (_pack_hostname(host), port)
    if 'host' in options and options['host'] is not None:
        headers.append('Host: %s' % options['host'])
    else:
        headers.append('Host: %s' % hostport)
    if 'suppress_origin' not in options or not options['suppress_origin']:
        if 'origin' in options and options['origin'] is not None:
            headers.append('Origin: %s' % options['origin'])
        else:
            headers.append('Origin: http://%s' % hostport)
    key = _create_sec_websocket_key()
    if not 'header' in options or 'Sec-WebSocket-Key' not in options['header']:
        key = _create_sec_websocket_key()
        headers.append('Sec-WebSocket-Key: %s' % key)
    else:
        key = options['header']['Sec-WebSocket-Key']
    if not 'header' in options or 'Sec-WebSocket-Version' not in options['header']:
        headers.append('Sec-WebSocket-Version: %s' % VERSION)
    subprotocols = options.get('subprotocols')
    if subprotocols:
        headers.append('Sec-WebSocket-Protocol: %s' % ','.join(subprotocols))
    if 'header' in options:
        header = options['header']
        if isinstance(header, dict):
            header = [': '.join([k, v]) for k, v in header.items() if v is not None]
        headers.extend(header)
    server_cookie = CookieJar.get(host)
    client_cookie = options.get('cookie', None)
    cookie = '; '.join(filter(None, [server_cookie, client_cookie]))
    if cookie:
        headers.append('Cookie: %s' % cookie)
    headers.append('')
    headers.append('')
    return (headers, key)