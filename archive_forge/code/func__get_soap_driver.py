import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _get_soap_driver(self):
    if pyvmomi is None:
        raise ImportError('Missing "pyvmomi" dependency. You can install it using pip - pip install pyvmomi')
    self.driver_soap = VSphereNodeDriver(self.host, self.username, self.connection.secret, ca_cert=self.connection.connection.ca_cert)