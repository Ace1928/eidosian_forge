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
def _strip_credentials(self, kwargs):
    if kwargs.get('body') and self.password:
        log_kwargs = kwargs.copy()
        log_kwargs['body'] = kwargs['body'].replace(self.password, 'REDACTED')
        return log_kwargs
    else:
        return kwargs