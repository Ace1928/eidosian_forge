import logging
import ssl
import typing
from socksio import socks5
from .._backends.sync import SyncBackend
from .._backends.base import NetworkBackend, NetworkStream
from .._exceptions import ConnectionNotAvailable, ProxyError
from .._models import URL, Origin, Request, Response, enforce_bytes, enforce_url
from .._ssl import default_ssl_context
from .._synchronization import Lock
from .._trace import Trace
from .connection_pool import ConnectionPool
from .http11 import HTTP11Connection
from .interfaces import ConnectionInterface
def _init_socks5_connection(stream: NetworkStream, *, host: bytes, port: int, auth: typing.Optional[typing.Tuple[bytes, bytes]]=None) -> None:
    conn = socks5.SOCKS5Connection()
    auth_method = socks5.SOCKS5AuthMethod.NO_AUTH_REQUIRED if auth is None else socks5.SOCKS5AuthMethod.USERNAME_PASSWORD
    conn.send(socks5.SOCKS5AuthMethodsRequest([auth_method]))
    outgoing_bytes = conn.data_to_send()
    stream.write(outgoing_bytes)
    incoming_bytes = stream.read(max_bytes=4096)
    response = conn.receive_data(incoming_bytes)
    assert isinstance(response, socks5.SOCKS5AuthReply)
    if response.method != auth_method:
        requested = AUTH_METHODS.get(auth_method, 'UNKNOWN')
        responded = AUTH_METHODS.get(response.method, 'UNKNOWN')
        raise ProxyError(f'Requested {requested} from proxy server, but got {responded}.')
    if response.method == socks5.SOCKS5AuthMethod.USERNAME_PASSWORD:
        assert auth is not None
        username, password = auth
        conn.send(socks5.SOCKS5UsernamePasswordRequest(username, password))
        outgoing_bytes = conn.data_to_send()
        stream.write(outgoing_bytes)
        incoming_bytes = stream.read(max_bytes=4096)
        response = conn.receive_data(incoming_bytes)
        assert isinstance(response, socks5.SOCKS5UsernamePasswordReply)
        if not response.success:
            raise ProxyError('Invalid username/password')
    conn.send(socks5.SOCKS5CommandRequest.from_address(socks5.SOCKS5Command.CONNECT, (host, port)))
    outgoing_bytes = conn.data_to_send()
    stream.write(outgoing_bytes)
    incoming_bytes = stream.read(max_bytes=4096)
    response = conn.receive_data(incoming_bytes)
    assert isinstance(response, socks5.SOCKS5Reply)
    if response.reply_code != socks5.SOCKS5ReplyCode.SUCCEEDED:
        reply_code = REPLY_CODES.get(response.reply_code, 'UNKOWN')
        raise ProxyError(f'Proxy Server could not connect: {reply_code}.')