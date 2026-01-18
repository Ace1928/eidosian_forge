import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
Retrieve unscoped token after authentcation with ADFS server.

        This is a multistep process:

        * Prepare ADFS Request Securty Token -
          build an etree.XML object filling certain attributes with proper user
          credentials, created/expires dates (ticket is be valid for 120
          seconds as currently we don't handle reusing ADFS issued security
          tokens).

        * Send ADFS Security token to the ADFS server. Step handled by

        * Receive and parse security token, extract actual SAML assertion and
          prepare a request addressed for the Service Provider endpoint.
          This also includes changing namespaces in the XML document. Step
          handled by ``ADFSPassword._prepare_sp_request()`` method.

        * Send prepared assertion to the Service Provider endpoint. Usually
          the server will respond with HTTP 301 code which should be ignored as
          the 'location' header doesn't contain protected area. The goal of
          this operation is fetching the session cookie which later allows for
          accessing protected URL endpoints. Step handed by
          ``ADFSPassword._send_assertion_to_service_provider()`` method.

        * Once the session cookie is issued, the protected endpoint can be
          accessed and an unscoped token can be retrieved. Step handled by
          ``ADFSPassword._access_service_provider()`` method.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: AccessInfo
        :rtype: :py:class:`keystoneauth1.access.AccessInfo`

        