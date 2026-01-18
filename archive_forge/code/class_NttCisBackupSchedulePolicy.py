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
class NttCisBackupSchedulePolicy:
    """
    A representation of a schedule policy
    """

    def __init__(self, name, description):
        """
        Initialize an instance of :class:`NttCisBackupSchedulePolicy`

        :param name: The name of the policy i.e 12AM - 6AM
        :type  name: ``str``

        :param description: Short summary of the details of the policy
        :type  description: ``str``
        """
        self.name = name
        self.description = description

    def __repr__(self):
        return '<NttCisBackupSchedulePolicy: name=%s>' % self.name