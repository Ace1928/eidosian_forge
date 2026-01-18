import hmac, base64, random, time, warnings
from functools import reduce
from paste.request import get_cookies
class AuthCookieSigner(object):
    """
    save/restore ``environ`` entries via digially signed cookie

    This class converts content into a timed and digitally signed
    cookie, as well as having the facility to reverse this procedure.
    If the cookie, after the content is encoded and signed exceeds the
    maximum length (4096), then CookieTooLarge exception is raised.

    The timeout of the cookie is handled on the server side for a few
    reasons.  First, if a 'Expires' directive is added to a cookie, then
    the cookie becomes persistent (lasting even after the browser window
    has closed). Second, the user's clock may be wrong (perhaps
    intentionally). The timeout is specified in minutes; and expiration
    date returned is rounded to one second.

    Constructor Arguments:

        ``secret``

            This is a secret key if you want to syncronize your keys so
            that the cookie will be good across a cluster of computers.
            It is recommended via the HMAC specification (RFC 2104) that
            the secret key be 64 bytes since this is the block size of
            the hashing.  If you do not provide a secret key, a random
            one is generated each time you create the handler; this
            should be sufficient for most cases.

        ``timeout``

            This is the time (in minutes) from which the cookie is set
            to expire.  Note that on each request a new (replacement)
            cookie is sent, hence this is effectively a session timeout
            parameter for your entire cluster.  If you do not provide a
            timeout, it is set at 30 minutes.

        ``maxlen``

            This is the maximum size of the *signed* cookie; hence the
            actual content signed will be somewhat less.  If the cookie
            goes over this size, a ``CookieTooLarge`` exception is
            raised so that unexpected handling of cookies on the client
            side are avoided.  By default this is set at 4k (4096 bytes),
            which is the standard cookie size limit.

    """

    def __init__(self, secret=None, timeout=None, maxlen=None):
        self.timeout = timeout or 30
        if isinstance(timeout, str):
            raise ValueError('Timeout must be a number (minutes), not a string (%r)' % timeout)
        self.maxlen = maxlen or 4096
        self.secret = secret or new_secret()

    def sign(self, content):
        """
        Sign the content returning a valid cookie (that does not
        need to be escaped and quoted).  The expiration of this
        cookie is handled server-side in the auth() function.
        """
        timestamp = make_time(time.time() + 60 * self.timeout)
        content = content.encode('utf8')
        timestamp = timestamp.encode('utf8')
        cookie = base64.encodebytes(hmac.new(self.secret, content, sha1).digest() + timestamp + content)
        cookie = cookie.replace(b'/', b'_').replace(b'=', b'~')
        cookie = cookie.replace(b'\n', b'').replace(b'\r', b'')
        if len(cookie) > self.maxlen:
            raise CookieTooLarge(content, cookie)
        return cookie

    def auth(self, cookie):
        """
        Authenticate the cooke using the signature, verify that it
        has not expired; and return the cookie's content
        """
        decode = base64.decodestring(cookie.replace('_', '/').replace('~', '='))
        signature = decode[:_signature_size]
        expires = decode[_signature_size:_header_size]
        content = decode[_header_size:]
        if signature == hmac.new(self.secret, content, sha1).digest():
            if int(expires) > int(make_time(time.time())):
                return content
            else:
                pass
        else:
            pass