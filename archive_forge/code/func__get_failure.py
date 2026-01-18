import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
def _get_failure(e):
    xpath = '/s:Envelope/s:Body/s:Fault/s:Code/s:Subcode/s:Value'
    content = e.response.content
    try:
        obj = self.str_to_xml(content).xpath(xpath, namespaces=self.NAMESPACES)
        obj = self._first(obj)
        return obj.text
    except (IndexError, exceptions.AuthorizationFailure):
        raise e