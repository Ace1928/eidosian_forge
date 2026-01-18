from urllib.parse import urlencode
from urllib.request import urlopen
from paste.request import construct_url
from paste.httpexceptions import HTTPSeeOther, HTTPForbidden
def AuthCASHandler(application, authority):
    """
    middleware to implement CAS 1.0 authentication

    There are several possible outcomes:

    0. If the REMOTE_USER environment variable is already populated;
       then this middleware is a no-op, and the request is passed along
       to the application.

    1. If a query argument 'ticket' is found, then an attempt to
       validate said ticket /w the authentication service done.  If the
       ticket is not validated; an 403 'Forbidden' exception is raised.
       Otherwise, the REMOTE_USER variable is set with the NetID that
       was validated and AUTH_TYPE is set to "cas".

    2. Otherwise, a 303 'See Other' is returned to the client directing
       them to login using the CAS service.  After logon, the service
       will send them back to this same URL, only with a 'ticket' query
       argument.

    Parameters:

        ``authority``

            This is a fully-qualified URL to a CAS 1.0 service. The URL
            should end with a '/' and have the 'login' and 'validate'
            sub-paths as described in the CAS 1.0 documentation.

    """
    assert authority.endswith('/') and authority.startswith('http')

    def cas_application(environ, start_response):
        username = environ.get('REMOTE_USER', '')
        if username:
            return application(environ, start_response)
        qs = environ.get('QUERY_STRING', '').split('&')
        if qs and qs[-1].startswith('ticket='):
            ticket = qs.pop().split('=', 1)[1]
            environ['QUERY_STRING'] = '&'.join(qs)
            service = construct_url(environ)
            args = urlencode({'service': service, 'ticket': ticket})
            requrl = authority + 'validate?' + args
            result = urlopen(requrl).read().split('\n')
            if 'yes' == result[0]:
                environ['REMOTE_USER'] = result[1]
                environ['AUTH_TYPE'] = 'cas'
                return application(environ, start_response)
            exce = CASLoginFailure()
        else:
            service = construct_url(environ)
            args = urlencode({'service': service})
            location = authority + 'login?' + args
            exce = CASAuthenticate(location)
        return exce.wsgi_application(environ, start_response)
    return cas_application