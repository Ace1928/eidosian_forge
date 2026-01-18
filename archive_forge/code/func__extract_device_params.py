from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
def _extract_device_params(kwds):
    device_params = kwds.pop('device_params', None)
    return device_params