import copy
import logging
import os
import platform
import socket
import warnings
import botocore.client
import botocore.configloader
import botocore.credentials
import botocore.tokens
from botocore import (
from botocore.compat import HAS_CRT, MutableMapping
from botocore.configprovider import (
from botocore.errorfactory import ClientExceptionsFactory
from botocore.exceptions import (
from botocore.hooks import (
from botocore.loaders import create_loader
from botocore.model import ServiceModel
from botocore.parsers import ResponseParserFactory
from botocore.regions import EndpointResolver
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.compat import HAS_CRT  # noqa
def _create_csm_monitor(self):
    if self.get_config_variable('csm_enabled'):
        client_id = self.get_config_variable('csm_client_id')
        host = self.get_config_variable('csm_host')
        port = self.get_config_variable('csm_port')
        handler = monitoring.Monitor(adapter=monitoring.MonitorEventAdapter(), publisher=monitoring.SocketPublisher(socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM), host=host, port=port, serializer=monitoring.CSMSerializer(csm_client_id=client_id)))
        return handler
    return None