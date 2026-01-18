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
@staticmethod
def _remove_admin_password(guest_customization_section):
    """
        Remove AdminPassword element from GuestCustomizationSection if it
        would be invalid to include it.

        This was originally done unconditionally due to an "API quirk" of
        unknown origin or effect. When AdminPasswordEnabled is set to true
        and AdminPasswordAuto is false, the admin password must be set or
        an error will ensue, and vice versa.
        :param guest_customization_section: GuestCustomizationSection element
                                            to remove password from (if valid
                                            to do so)
        :type guest_customization_section: ``ET.Element``
        """
    admin_pass_enabled = guest_customization_section.find(fixxpath(guest_customization_section, 'AdminPasswordEnabled'))
    admin_pass_auto = guest_customization_section.find(fixxpath(guest_customization_section, 'AdminPasswordAuto'))
    admin_pass = guest_customization_section.find(fixxpath(guest_customization_section, 'AdminPassword'))
    if admin_pass is not None and (admin_pass_enabled is None or admin_pass_enabled.text != 'true' or admin_pass_auto is None or (admin_pass_auto.text != 'false')):
        guest_customization_section.remove(admin_pass)