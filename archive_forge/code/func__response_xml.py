import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
def _response_xml(response, name):
    try:
        return etree.XML(response.content)
    except etree.XMLSyntaxError as e:
        msg = 'SAML2: Error parsing XML returned from %s: %s' % (name, e)
        raise InvalidResponse(msg)