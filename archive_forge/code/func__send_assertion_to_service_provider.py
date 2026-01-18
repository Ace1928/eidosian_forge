import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
def _send_assertion_to_service_provider(self, session):
    """Send prepared assertion to a service provider.

        As the assertion doesn't contain a protected resource, the value from
        the ``location`` header is not valid and we should not let the Session
        object get redirected there. The aim of this call is to get a cookie in
        the response which is required for entering a protected endpoint.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :raises: Corresponding HTTP error exception

        """
    session.post(url=self.service_provider_endpoint, data=self.encoded_assertion, headers=self.HEADER_X_FORM, redirect=False, authenticated=False)