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
class NttCisDefaultiRule:
    """
    A default iRule for a network domain, can be applied to a listener
    """

    def __init__(self, id, name, compatible_listeners):
        """
        Initialize an instance of :class:`NttCisefaultiRule`

        :param id: The ID of the iRule
        :type  id: ``str``

        :param name: The name of the iRule
        :type  name: ``str``

        :param compatible_listeners: List of compatible Virtual Listener types
        :type  compatible_listeners: ``list`` of
            :class:`NttCisVirtualListenerCompatibility`
        """
        self.id = id
        self.name = name
        self.compatible_listeners = compatible_listeners

    def __repr__(self):
        return '<NttCisDefaultiRule: id=%s, name=%s>' % (self.id, self.name)