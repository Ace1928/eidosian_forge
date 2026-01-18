import time
import hashlib
from libcloud.utils.py3 import b
from libcloud.common.base import ConnectionKey
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
class GandiResponse(XMLRPCResponse):
    """
    A Base Gandi Response class to derive from.
    """