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
class ControlAccess:
    """
    Represents control access settings of a node
    """

    class AccessLevel:
        READ_ONLY = 'ReadOnly'
        CHANGE = 'Change'
        FULL_CONTROL = 'FullControl'

    def __init__(self, node, everyone_access_level, subjects=None):
        self.node = node
        self.everyone_access_level = everyone_access_level
        if not subjects:
            subjects = []
        self.subjects = subjects

    def __repr__(self):
        return '<ControlAccess: node=%s, everyone_access_level=%s, subjects=%s>' % (self.node, self.everyone_access_level, self.subjects)