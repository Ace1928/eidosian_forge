import logging
import os
import debtcollector.renames
from keystoneauth1 import access
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
import requests
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def _authenticate_noauth(self):
    if not self.endpoint_url:
        message = _('For "noauth" authentication strategy, the endpoint must be specified either in the constructor or using --os-url')
        raise exceptions.Unauthorized(message=message)