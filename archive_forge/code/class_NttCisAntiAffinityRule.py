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
class NttCisAntiAffinityRule:
    """
    Anti-Affinity rule for NTTCIS

    An Anti-Affinity rule ensures that servers in the rule will
    not reside on the same VMware ESX host.
    """

    def __init__(self, id, node_list):
        """
        Instantiate a new :class:`NttCisDataAntiAffinityRule`

        :param id: The ID of the Anti-Affinity rule
        :type  id: ``str``

        :param node_list: List of node ids that belong in this rule
        :type  node_list: ``list`` of ``str``
        """
        self.id = id
        self.node_list = node_list

    def __repr__(self):
        return '<NttCisAntiAffinityRule: id=%s>' % self.id