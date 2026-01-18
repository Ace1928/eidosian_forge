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
def _set_soap_headers(self, op_id):
    """Set SOAP headers for the next remote call to vCenter.

        SOAP headers may include operation ID and vcSessionCookie.
        The operation ID is a random string which allows to correlate log
        messages across different systems (OpenStack, vCenter, ESX).
        vcSessionCookie is needed when making PBM calls.
        """
    headers = []
    if self._vc_session_cookie:
        elem = element.Element('vcSessionCookie').setText(self._vc_session_cookie)
        headers.append(elem)
    if op_id:
        elem = element.Element('operationID').setText(op_id)
        headers.append(elem)
    if headers:
        self.client.set_options(soapheaders=headers)