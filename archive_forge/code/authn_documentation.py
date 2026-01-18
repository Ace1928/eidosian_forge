import logging
import time
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit
from saml2 import SAMLError
import saml2.cryptography.symmetric
from saml2.httputil import Redirect
from saml2.httputil import Response
from saml2.httputil import Unauthorized
from saml2.httputil import make_cookie
from saml2.httputil import parse_cookie

            Verifies the username and password agains a LDAP server
            :param pwd: The password
            :param user: The username
            :return: AssertionError if the LDAP verification failed.
            