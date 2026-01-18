import copy
import logging
import socket
import botocore.exceptions
import botocore.parsers
import botocore.serialize
from botocore.config import Config
from botocore.endpoint import EndpointCreator
from botocore.regions import EndpointResolverBuiltins as EPRBuiltins
from botocore.regions import EndpointRulesetResolver
from botocore.signers import RequestSigner
from botocore.useragent import UserAgentString
from botocore.utils import ensure_boolean, is_s3_accelerate_url
def _compute_socket_options(self, scoped_config, client_config=None):
    socket_options = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
    client_keepalive = client_config and client_config.tcp_keepalive
    scoped_keepalive = scoped_config and self._ensure_boolean(scoped_config.get('tcp_keepalive', False))
    if client_keepalive or scoped_keepalive:
        socket_options.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
    return socket_options