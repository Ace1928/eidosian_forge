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
class VCloud_5_1_NodeDriver(VCloud_1_5_NodeDriver):

    @staticmethod
    def _validate_vm_memory(vm_memory):
        if vm_memory is None:
            return None
        elif vm_memory % 4 != 0:
            raise ValueError('%s is not a valid vApp VM memory value' % vm_memory)