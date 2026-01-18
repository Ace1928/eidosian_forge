from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
def _extract_nc_params(kwds):
    nc_params = kwds.pop('nc_params', {})
    return nc_params