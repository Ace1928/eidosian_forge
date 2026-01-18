import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
class AzureNodeLocation(NodeLocation):

    def __init__(self, id, name, country, driver, available_services, virtual_machine_role_sizes):
        super().__init__(id, name, country, driver)
        self.available_services = available_services
        self.virtual_machine_role_sizes = virtual_machine_role_sizes

    def __repr__(self):
        return '<AzureNodeLocation: id=%s, name=%s, country=%s, driver=%s services=%s virtualMachineRoleSizes=%s >' % (self.id, self.name, self.country, self.driver.name, ','.join(self.available_services), ','.join(self.virtual_machine_role_sizes))