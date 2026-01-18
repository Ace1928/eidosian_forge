import http.client as http
import urllib.parse as urlparse
import httplib2
from keystoneclient import service_catalog as ks_service_catalog
from oslo_serialization import jsonutils
from glance.common import exception
from glance.i18n import _
class NoAuthStrategy(BaseStrategy):

    def authenticate(self):
        pass

    @property
    def is_authenticated(self):
        return True

    @property
    def strategy(self):
        return 'noauth'