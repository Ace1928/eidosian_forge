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
def _validate_vm_script(vm_script):
    if vm_script is None:
        return
    if not os.path.isabs(vm_script):
        vm_script = os.path.expanduser(vm_script)
        vm_script = os.path.abspath(vm_script)
    if not os.path.isfile(vm_script):
        raise LibcloudError('%s the VM script file does not exist' % vm_script)
    try:
        open(vm_script).read()
    except Exception:
        raise
    return vm_script