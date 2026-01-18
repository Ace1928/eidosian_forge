import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
def _handle_hybi_request(self, environ):
    if 'eventlet.input' in environ:
        sock = environ['eventlet.input'].get_socket()
    elif 'gunicorn.socket' in environ:
        sock = environ['gunicorn.socket']
    else:
        raise Exception('No eventlet.input or gunicorn.socket present in environ.')
    hybi_version = environ['HTTP_SEC_WEBSOCKET_VERSION']
    if hybi_version not in ('8', '13'):
        raise BadRequest(status='426 Upgrade Required', headers=[('Sec-WebSocket-Version', '8, 13')])
    self.protocol_version = int(hybi_version)
    if 'HTTP_SEC_WEBSOCKET_KEY' not in environ:
        raise BadRequest()
    origin = environ.get('HTTP_ORIGIN', environ.get('HTTP_SEC_WEBSOCKET_ORIGIN', '') if self.protocol_version <= 8 else '')
    if self.origin_checker is not None:
        if not self.origin_checker(environ.get('HTTP_HOST'), origin):
            raise BadRequest(status='403 Forbidden')
    protocols = environ.get('HTTP_SEC_WEBSOCKET_PROTOCOL', None)
    negotiated_protocol = None
    if protocols:
        for p in (i.strip() for i in protocols.split(',')):
            if p in self.supported_protocols:
                negotiated_protocol = p
                break
    key = environ['HTTP_SEC_WEBSOCKET_KEY']
    response = base64.b64encode(sha1(key.encode() + PROTOCOL_GUID).digest())
    handshake_reply = [b'HTTP/1.1 101 Switching Protocols', b'Upgrade: websocket', b'Connection: Upgrade', b'Sec-WebSocket-Accept: ' + response]
    if negotiated_protocol:
        handshake_reply.append(b'Sec-WebSocket-Protocol: ' + negotiated_protocol.encode())
    parsed_extensions = {}
    extensions = self._parse_extension_header(environ.get('HTTP_SEC_WEBSOCKET_EXTENSIONS'))
    deflate = self._negotiate_permessage_deflate(extensions)
    if deflate is not None:
        parsed_extensions['permessage-deflate'] = deflate
    formatted_ext = self._format_extension_header(parsed_extensions)
    if formatted_ext is not None:
        handshake_reply.append(b'Sec-WebSocket-Extensions: ' + formatted_ext)
    sock.sendall(b'\r\n'.join(handshake_reply) + b'\r\n\r\n')
    return RFC6455WebSocket(sock, environ, self.protocol_version, protocol=negotiated_protocol, extensions=parsed_extensions, max_frame_length=self.max_frame_length)