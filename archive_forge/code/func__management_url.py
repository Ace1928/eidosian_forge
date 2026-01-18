import http.client as http
import urllib.parse as urlparse
import httplib2
from keystoneclient import service_catalog as ks_service_catalog
from oslo_serialization import jsonutils
from glance.common import exception
from glance.i18n import _
def _management_url(self, resp):
    for url_header in ('x-image-management-url', 'x-server-management-url', 'x-glance'):
        try:
            return resp[url_header]
        except KeyError as e:
            not_found = e
    raise not_found