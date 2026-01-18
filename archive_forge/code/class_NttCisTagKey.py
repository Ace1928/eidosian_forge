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
class NttCisTagKey:
    """
    A representation of a Tag Key in NTTCIS
    A tag key is required to tag an asset
    """

    def __init__(self, id, name, description, value_required, display_on_report):
        """
        Initialize an instance of :class:`NttCisTagKey`

        :param id: GUID of the tag key
        :type  id: ``str``

        :param name: Name of the tag key
        :type  name: ``str``

        :param description: Description of the tag key
        :type  description: ``str``

        :param value_required: If a value is required for this tag key
        :type  value_required: ``bool``

        :param display_on_report: If this tag key should be displayed on
                                  usage reports
        :type  display_on_report: ``bool``
        """
        self.id = id
        self.name = name
        self.description = description
        self.value_required = value_required
        self.display_on_report = display_on_report

    def __repr__(self):
        return 'NttCisTagKey: id=%s name=%s>' % (self.id, self.name)