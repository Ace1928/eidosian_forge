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
def _get_result_tty(self, stream, res, is_tty):
    if is_tty:
        return self._stream_raw_result(res) if stream else self._result(res, binary=True)
    self._raise_for_status(res)
    sep = b''
    if stream:
        return self._multiplexed_response_stream_helper(res)
    else:
        return sep.join([x for x in self._multiplexed_buffer_helper(res)])