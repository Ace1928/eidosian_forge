import json
import struct
import urllib
from functools import partial
import requests
import requests.exceptions
import websocket
from .. import auth
from ..constants import (DEFAULT_NUM_POOLS, DEFAULT_NUM_POOLS_SSH,
from ..errors import (DockerException, InvalidVersion, TLSParameterError,
from ..tls import TLSConfig
from ..transport import SSLHTTPAdapter, UnixHTTPAdapter
from ..utils import check_resource, config, update_headers, utils
from ..utils.json_stream import json_stream
from ..utils.proxy import ProxyConfig
from ..utils.socket import consume_socket_output, demux_adaptor, frames_iter
from .build import BuildApiMixin
from .config import ConfigApiMixin
from .container import ContainerApiMixin
from .daemon import DaemonApiMixin
from .exec_api import ExecApiMixin
from .image import ImageApiMixin
from .network import NetworkApiMixin
from .plugin import PluginApiMixin
from .secret import SecretApiMixin
from .service import ServiceApiMixin
from .swarm import SwarmApiMixin
from .volume import VolumeApiMixin
def _stream_helper(self, response, decode=False):
    """Generator for data coming from a chunked-encoded HTTP response."""
    if response.raw._fp.chunked:
        if decode:
            yield from json_stream(self._stream_helper(response, False))
        else:
            reader = response.raw
            while not reader.closed:
                data = reader.read(1)
                if not data:
                    break
                if reader._fp.chunk_left:
                    data += reader.read(reader._fp.chunk_left)
                yield data
    else:
        yield self._result(response, json=decode)