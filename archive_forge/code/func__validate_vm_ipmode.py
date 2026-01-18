import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
@staticmethod
def _validate_vm_ipmode(vm_ipmode):
    if vm_ipmode is None:
        return
    elif vm_ipmode == 'MANUAL':
        raise NotImplementedError('MANUAL IP mode: The interface for supplying IPAddress does not exist yet')
    elif vm_ipmode not in IP_MODE_VALS_1_5:
        raise ValueError('%s is not a valid IP address allocation mode value' % vm_ipmode)