from __future__ import annotations
from collections import deque
from functools import partial
from io import BytesIO
from time import time
from kombu.asynchronous.hub import READ, WRITE, Hub, get_event_loop
from kombu.exceptions import HttpError
from kombu.utils.encoding import bytes_to_str
from .base import BaseClient
def _setup_request(self, curl, request, buffer, headers, _pycurl=pycurl):
    setopt = curl.setopt
    setopt(_pycurl.URL, bytes_to_str(request.url))
    request.headers.setdefault('Expect', '')
    request.headers.setdefault('Pragma', '')
    setopt(_pycurl.HTTPHEADER, ['{}: {}'.format(*h) for h in request.headers.items()])
    setopt(_pycurl.HEADERFUNCTION, partial(request.on_header or self.on_header, request.headers))
    setopt(_pycurl.WRITEFUNCTION, request.on_stream or buffer.write)
    setopt(_pycurl.FOLLOWLOCATION, request.follow_redirects)
    setopt(_pycurl.USERAGENT, bytes_to_str(request.user_agent or DEFAULT_USER_AGENT))
    if request.network_interface:
        setopt(_pycurl.INTERFACE, request.network_interface)
    setopt(_pycurl.ENCODING, 'gzip,deflate' if request.use_gzip else 'none')
    if request.proxy_host:
        if not request.proxy_port:
            raise ValueError('Request with proxy_host but no proxy_port')
        setopt(_pycurl.PROXY, request.proxy_host)
        setopt(_pycurl.PROXYPORT, request.proxy_port)
        if request.proxy_username:
            setopt(_pycurl.PROXYUSERPWD, '{}:{}'.format(request.proxy_username, request.proxy_password or ''))
    setopt(_pycurl.SSL_VERIFYPEER, 1 if request.validate_cert else 0)
    setopt(_pycurl.SSL_VERIFYHOST, 2 if request.validate_cert else 0)
    if request.ca_certs is not None:
        setopt(_pycurl.CAINFO, request.ca_certs)
    setopt(_pycurl.IPRESOLVE, pycurl.IPRESOLVE_WHATEVER)
    for meth in METH_TO_CURL.values():
        setopt(meth, False)
    try:
        meth = METH_TO_CURL[request.method]
    except KeyError:
        curl.setopt(_pycurl.CUSTOMREQUEST, request.method)
    else:
        curl.unsetopt(_pycurl.CUSTOMREQUEST)
        setopt(meth, True)
    if request.method in ('POST', 'PUT'):
        body = request.body.encode('utf-8') if request.body else b''
        reqbuffer = BytesIO(body)
        setopt(_pycurl.READFUNCTION, reqbuffer.read)
        if request.method == 'POST':

            def ioctl(cmd):
                if cmd == _pycurl.IOCMD_RESTARTREAD:
                    reqbuffer.seek(0)
            setopt(_pycurl.IOCTLFUNCTION, ioctl)
            setopt(_pycurl.POSTFIELDSIZE, len(body))
        else:
            setopt(_pycurl.INFILESIZE, len(body))
    elif request.method == 'GET':
        assert not request.body
    if request.auth_username is not None:
        auth_mode = {'basic': _pycurl.HTTPAUTH_BASIC, 'digest': _pycurl.HTTPAUTH_DIGEST}[request.auth_mode or 'basic']
        setopt(_pycurl.HTTPAUTH, auth_mode)
        userpwd = '{}:{}'.format(request.auth_username, request.auth_password or '')
        setopt(_pycurl.USERPWD, userpwd)
    else:
        curl.unsetopt(_pycurl.USERPWD)
    if request.client_cert is not None:
        setopt(_pycurl.SSLCERT, request.client_cert)
    if request.client_key is not None:
        setopt(_pycurl.SSLKEY, request.client_key)
    if request.on_prepare is not None:
        request.on_prepare(curl)