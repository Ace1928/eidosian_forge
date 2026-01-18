from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
import time, random
from urllib.parse import quote as url_quote
class AuthDigestHandler(object):
    """
    middleware for HTTP Digest authentication (RFC 2617)

    This component follows the procedure below:

        0. If the REMOTE_USER environment variable is already populated;
           then this middleware is a no-op, and the request is passed
           along to the application.

        1. If the HTTP_AUTHORIZATION header was not provided or specifies
           an algorithem other than ``digest``, then a HTTPUnauthorized
           response is generated with the challenge.

        2. If the response is malformed or or if the user's credientials
           do not pass muster, another HTTPUnauthorized is raised.

        3. If all goes well, and the user's credintials pass; then
           REMOTE_USER environment variable is filled in and the
           AUTH_TYPE is listed as 'digest'.

    Parameters:

        ``application``

            The application object is called only upon successful
            authentication, and can assume ``environ['REMOTE_USER']``
            is set.  If the ``REMOTE_USER`` is already set, this
            middleware is simply pass-through.

        ``realm``

            This is a identifier for the authority that is requesting
            authorization.  It is shown to the user and should be unique
            within the domain it is being used.

        ``authfunc``

            This is a callback function which performs the actual
            authentication; the signature of this callback is:

              authfunc(environ, realm, username) -> hashcode

            This module provides a 'digest_password' helper function
            which can help construct the hashcode; it is recommended
            that the hashcode is stored in a database, not the user's
            actual password (since you only need the hashcode).
    """

    def __init__(self, application, realm, authfunc):
        self.authenticate = AuthDigestAuthenticator(realm, authfunc)
        self.application = application

    def __call__(self, environ, start_response):
        username = REMOTE_USER(environ)
        if not username:
            result = self.authenticate(environ)
            if isinstance(result, str):
                AUTH_TYPE.update(environ, 'digest')
                REMOTE_USER.update(environ, result)
            else:
                return result.wsgi_application(environ, start_response)
        return self.application(environ, start_response)