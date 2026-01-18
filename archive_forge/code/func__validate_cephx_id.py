import collections
import ipaddress
from oslo_utils import uuidutils
import re
import string
from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient.v2 import share_instances
@staticmethod
def _validate_cephx_id(cephx_id):
    if not cephx_id:
        raise exceptions.CommandError('Ceph IDs may not be empty.')
    if not set(cephx_id) <= set(string.printable):
        raise exceptions.CommandError('Ceph IDs must consist of ASCII printable characters.')
    if '.' in cephx_id:
        raise exceptions.CommandError('Ceph IDs may not contain periods.')