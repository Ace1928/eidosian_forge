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
class Vdc:
    """
    Virtual datacenter (vDC) representation
    """

    def __init__(self, id, name, driver, allocation_model=None, cpu=None, memory=None, storage=None):
        self.id = id
        self.name = name
        self.driver = driver
        self.allocation_model = allocation_model
        self.cpu = cpu
        self.memory = memory
        self.storage = storage

    def __repr__(self):
        return '<Vdc: id={}, name={}, driver={}  ...>'.format(self.id, self.name, self.driver.name)