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
class NttCisScsiController:
    """
    A class that represents the disk on a server
    """

    def __init__(self, id, adapter_type, bus_number, state):
        """
        Instantiate a new :class:`DimensionDataServerDisk`

        :param id: The id of the controller
        :type  id: ``str``

        :param adapter_type: The 'brand' of adapter
        :type  adapter_type: ``str``

        :param bus_number: The bus number occupied on the virtual hardware
        :type  bus_nubmer: ``str``

        :param state: Current state (i.e. NORMAL)
        :type  speed: ``str``

        :param state: State of the disk (i.e. PENDING)
        :type  state: ``str``
        """
        self.id = id
        self.adapter_type = adapter_type
        self.bus_number = bus_number
        self.state = state

    def __repr__(self):
        return '<NttCisScsiController: id=%s, adapter_type=%s, bus_number=%s, state=%s' % (self.id, self.adapter_type, self.bus_number, self.state)