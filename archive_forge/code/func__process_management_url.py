import importlib.metadata
import logging
import warnings
from debtcollector import removals
from debtcollector import renames
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import packaging.version
import requests
from keystoneclient import _discover
from keystoneclient import access
from keystoneclient.auth import base
from keystoneclient import baseclient
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
def _process_management_url(self, region_name):
    try:
        self._management_url = self.auth_ref.service_catalog.url_for(service_type='identity', endpoint_type='admin', region_name=region_name)
    except exceptions.EndpointNotFound as e:
        _logger.debug('Failed to find endpoint for management url %s', e)