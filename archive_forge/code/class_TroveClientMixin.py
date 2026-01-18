import logging
from keystoneauth1 import adapter
from oslo_utils import importutils
import requests
from urllib import parse as urlparse
from troveclient.apiclient import client
from troveclient import exceptions
from troveclient import service_catalog
class TroveClientMixin(object):

    def get_database_api_version_from_endpoint(self):
        magic_tuple = urlparse.urlsplit(self.management_url)
        scheme, netloc, path, query, frag = magic_tuple
        v = path.split('/')[1]
        valid_versions = ['v1.0']
        if v not in valid_versions:
            msg = "Invalid client version '%s'. must be one of: %s" % (v, ', '.join(valid_versions))
            raise exceptions.UnsupportedVersion(msg)
        return v[1:]