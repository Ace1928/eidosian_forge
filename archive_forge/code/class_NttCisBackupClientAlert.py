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
class NttCisBackupClientAlert:
    """
    An alert for a backup client
    """

    def __init__(self, trigger, notify_list=[]):
        """
        Initialize an instance of :class:`NttCisBackupClientAlert`

        :param trigger: Trigger type for the client i.e. ON_FAILURE
        :type  trigger: ``str``

        :param notify_list: List of email addresses that are notified
                            when the alert is fired
        :type  notify_list: ``list`` of ``str``
        """
        self.trigger = trigger
        self.notify_list = notify_list

    def __repr__(self):
        return '<NttCisBackupClientAlert: trigger=%s>' % self.trigger