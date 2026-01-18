from base64 import b64decode
from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
class AuthBasicHandler(object):
    """
    HTTP/1.0 ``Basic`` authentication middleware

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

            This is a mandatory user-defined function which takes a
            ``environ``, ``username`` and ``password`` for its first
            three arguments.  It should return ``True`` if the user is
            authenticated.

    """

    def __init__(self, application, realm, authfunc):
        self.application = application
        self.authenticate = AuthBasicAuthenticator(realm, authfunc)

    def __call__(self, environ, start_response):
        username = REMOTE_USER(environ)
        if not username:
            result = self.authenticate(environ)
            if isinstance(result, str):
                AUTH_TYPE.update(environ, 'basic')
                REMOTE_USER.update(environ, result)
            else:
                return result.wsgi_application(environ, start_response)
        return self.application(environ, start_response)