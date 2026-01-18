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
def _validate_vm_names(names):
    if names is None:
        return
    hname_re = re.compile('^(([a-zA-Z]|[a-zA-Z][a-zA-Z0-9]*)[\\-])*([A-Za-z]|[A-Za-z][A-Za-z0-9]*[A-Za-z0-9])$')
    for name in names:
        if len(name) > 15:
            raise ValueError('The VM name "' + name + '" is too long for the computer name (max 15 chars allowed).')
        if not hname_re.match(name):
            raise ValueError('The VM name "' + name + '" can not be used. "' + name + '" is not a valid computer name for the VM.')