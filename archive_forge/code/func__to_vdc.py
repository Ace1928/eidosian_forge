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
def _to_vdc(self, vdc_elm):

    def get_capacity_values(capacity_elm):
        if capacity_elm is None:
            return None
        limit = int(capacity_elm.findtext(fixxpath(capacity_elm, 'Limit')))
        used = int(capacity_elm.findtext(fixxpath(capacity_elm, 'Used')))
        units = capacity_elm.findtext(fixxpath(capacity_elm, 'Units'))
        return Capacity(limit, used, units)
    cpu = get_capacity_values(vdc_elm.find(fixxpath(vdc_elm, 'ComputeCapacity/Cpu')))
    memory = get_capacity_values(vdc_elm.find(fixxpath(vdc_elm, 'ComputeCapacity/Memory')))
    storage = get_capacity_values(vdc_elm.find(fixxpath(vdc_elm, 'StorageCapacity')))
    return Vdc(id=vdc_elm.get('href'), name=vdc_elm.get('name'), driver=self, allocation_model=vdc_elm.findtext(fixxpath(vdc_elm, 'AllocationModel')), cpu=cpu, memory=memory, storage=storage)