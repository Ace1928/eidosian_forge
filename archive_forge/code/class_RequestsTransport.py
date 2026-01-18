import http.client as httplib
import io
import logging
import netaddr
from oslo_utils import timeutils
from oslo_utils import uuidutils
import requests
import suds
from suds import cache
from suds import client
from suds import plugin
import suds.sax.element as element
from suds import transport
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
class RequestsTransport(transport.Transport):

    def __init__(self, cacert=None, insecure=True, pool_maxsize=10, connection_timeout=None):
        transport.Transport.__init__(self)
        self.verify = cacert if cacert else not insecure
        self.session = requests.Session()
        self.session.mount('file:///', LocalFileAdapter(pool_maxsize=pool_maxsize))
        self.session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize))
        self.cookiejar = self.session.cookies
        self._connection_timeout = connection_timeout

    def open(self, request):
        resp = self.session.get(request.url, verify=self.verify)
        return io.BytesIO(resp.content)

    def send(self, request):
        resp = self.session.post(request.url, data=request.message, headers=request.headers, verify=self.verify, timeout=self._connection_timeout)
        return transport.Reply(resp.status_code, resp.headers, resp.content)