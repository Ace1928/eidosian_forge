import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class NttCisVirtualListenerCompatibility:
    """
    A compatibility preference for a persistence profile or iRule
    specifies which virtual listener types this profile or iRule can be
    applied to.
    """

    def __init__(self, type, protocol):
        self.type = type
        self.protocol = protocol

    def __repr__(self):
        return '<NttCisVirtualListenerCompatibility: type=%s, protocol=%s>' % (self.type, self.protocol)