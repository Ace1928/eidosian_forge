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
@check_resource('container')
def _check_is_tty(self, container):
    cont = self.inspect_container(container)
    return cont['Config']['Tty']