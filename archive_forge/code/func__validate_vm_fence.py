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
def _validate_vm_fence(vm_fence):
    if vm_fence is None:
        return
    elif vm_fence not in FENCE_MODE_VALS_1_5:
        raise ValueError('%s is not a valid fencing mode value' % vm_fence)