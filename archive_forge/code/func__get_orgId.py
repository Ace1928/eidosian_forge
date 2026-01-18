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
def _get_orgId(self):
    """
        Send the /myaccount API request to NTTC-CIS cloud and parse the
        'orgId' from the XML response object. We need the orgId to use most
        of the other API functions
        """
    if self._orgId is None:
        body = self.request_api_1('myaccount').object
        self._orgId = findtext(body, 'orgId', DIRECTORY_NS)
    return self._orgId