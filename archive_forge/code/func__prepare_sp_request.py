import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
def _prepare_sp_request(self):
    """Prepare ADFS Security Token to be sent to the Service Provider.

        The method works as follows:
        * Extract SAML2 assertion from the ADFS Security Token.
        * Replace namespaces
        * urlencode assertion
        * concatenate static string with the encoded assertion

        """
    assertion = self.adfs_token.xpath(self.ADFS_ASSERTION_XPATH, namespaces=self.ADFS_TOKEN_NAMESPACES)
    assertion = self._first(assertion)
    assertion = self.xml_to_str(assertion)
    assertion = assertion.replace(b'http://docs.oasis-open.org/ws-sx/ws-trust/200512', b'http://schemas.xmlsoap.org/ws/2005/02/trust')
    encoded_assertion = urllib.parse.quote(assertion)
    self.encoded_assertion = 'wa=wsignin1.0&wresult=' + encoded_assertion