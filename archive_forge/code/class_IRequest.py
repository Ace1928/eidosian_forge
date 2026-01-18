from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class IRequest(Interface):
    """
    An HTTP request.

    @since: 9.0
    """
    method = Attribute('A L{bytes} giving the HTTP method that was used.')
    uri = Attribute('A L{bytes} giving the full encoded URI which was requested (including query arguments).')
    path = Attribute('A L{bytes} giving the encoded query path of the request URI (not including query arguments).')
    args = Attribute("A mapping of decoded query argument names as L{bytes} to corresponding query argument values as L{list}s of L{bytes}.  For example, for a URI with C{foo=bar&foo=baz&quux=spam} for its query part, C{args} will be C{{b'foo': [b'bar', b'baz'], b'quux': [b'spam']}}.")
    prepath = Attribute('The URL path segments which have been processed during resource traversal, as a list of L{bytes}.')
    postpath = Attribute('The URL path segments which have not (yet) been processed during resource traversal, as a list of L{bytes}.')
    requestHeaders = Attribute('A L{http_headers.Headers} instance giving all received HTTP request headers.')
    content = Attribute('A file-like object giving the request body.  This may be a file on disk, an L{io.BytesIO}, or some other type.  The implementation is free to decide on a per-request basis.')
    responseHeaders = Attribute('A L{http_headers.Headers} instance holding all HTTP response headers to be sent.')

    def getHeader(key):
        """
        Get an HTTP request header.

        @type key: L{bytes} or L{str}
        @param key: The name of the header to get the value of.

        @rtype: L{bytes} or L{str} or L{None}
        @return: The value of the specified header, or L{None} if that header
            was not present in the request. The string type of the result
            matches the type of C{key}.
        """

    def getCookie(key):
        """
        Get a cookie that was sent from the network.

        @type key: L{bytes}
        @param key: The name of the cookie to get.

        @rtype: L{bytes} or L{None}
        @returns: The value of the specified cookie, or L{None} if that cookie
            was not present in the request.
        """

    def getAllHeaders():
        """
        Return dictionary mapping the names of all received headers to the last
        value received for each.

        Since this method does not return all header information,
        C{requestHeaders.getAllRawHeaders()} may be preferred.
        """

    def getRequestHostname():
        """
        Get the hostname that the HTTP client passed in to the request.

        This will either use the C{Host:} header (if it is available; which,
        for a spec-compliant request, it will be) or the IP address of the host
        we are listening on if the header is unavailable.

        @note: This is the I{host portion} of the requested resource, which
            means that:

                1. it might be an IPv4 or IPv6 address, not just a DNS host
                   name,

                2. there's no guarantee it's even a I{valid} host name or IP
                   address, since the C{Host:} header may be malformed,

                3. it does not include the port number.

        @returns: the requested hostname

        @rtype: L{bytes}
        """

    def getHost():
        """
        Get my originally requesting transport's host.

        @return: An L{IAddress<twisted.internet.interfaces.IAddress>}.
        """

    def getClientAddress():
        """
        Return the address of the client who submitted this request.

        The address may not be a network address.  Callers must check
        its type before using it.

        @since: 18.4

        @return: the client's address.
        @rtype: an L{IAddress} provider.
        """

    def getClientIP():
        """
        Return the IP address of the client who submitted this request.

        This method is B{deprecated}.  See L{getClientAddress} instead.

        @returns: the client IP address or L{None} if the request was submitted
            over a transport where IP addresses do not make sense.
        @rtype: L{str} or L{None}
        """

    def getUser():
        """
        Return the HTTP user sent with this request, if any.

        If no user was supplied, return the empty string.

        @returns: the HTTP user, if any
        @rtype: L{str}
        """

    def getPassword():
        """
        Return the HTTP password sent with this request, if any.

        If no password was supplied, return the empty string.

        @returns: the HTTP password, if any
        @rtype: L{str}
        """

    def isSecure():
        """
        Return True if this request is using a secure transport.

        Normally this method returns True if this request's HTTPChannel
        instance is using a transport that implements ISSLTransport.

        This will also return True if setHost() has been called
        with ssl=True.

        @returns: True if this request is secure
        @rtype: C{bool}
        """

    def getSession(sessionInterface=None):
        """
        Look up the session associated with this request or create a new one if
        there is not one.

        @return: The L{Session} instance identified by the session cookie in
            the request, or the C{sessionInterface} component of that session
            if C{sessionInterface} is specified.
        """

    def URLPath():
        """
        @return: A L{URLPath<twisted.python.urlpath.URLPath>} instance
            which identifies the URL for which this request is.
        """

    def prePathURL():
        """
        At any time during resource traversal or resource rendering,
        returns an absolute URL to the most nested resource which has
        yet been reached.

        @see: {twisted.web.server.Request.prepath}

        @return: An absolute URL.
        @rtype: L{bytes}
        """

    def rememberRootURL():
        """
        Remember the currently-processed part of the URL for later
        recalling.
        """

    def getRootURL():
        """
        Get a previously-remembered URL.

        @return: An absolute URL.
        @rtype: L{bytes}
        """

    def finish():
        """
        Indicate that the response to this request is complete.
        """

    def write(data):
        """
        Write some data to the body of the response to this request.  Response
        headers are written the first time this method is called, after which
        new response headers may not be added.

        @param data: Bytes of the response body.
        @type data: L{bytes}
        """

    def addCookie(k, v, expires=None, domain=None, path=None, max_age=None, comment=None, secure=None):
        """
        Set an outgoing HTTP cookie.

        In general, you should consider using sessions instead of cookies, see
        L{twisted.web.server.Request.getSession} and the
        L{twisted.web.server.Session} class for details.
        """

    def setResponseCode(code, message=None):
        """
        Set the HTTP response code.

        @type code: L{int}
        @type message: L{bytes}
        """

    def setHeader(k, v):
        """
        Set an HTTP response header.  Overrides any previously set values for
        this header.

        @type k: L{bytes} or L{str}
        @param k: The name of the header for which to set the value.

        @type v: L{bytes} or L{str}
        @param v: The value to set for the named header. A L{str} will be
            UTF-8 encoded, which may not interoperable with other
            implementations. Avoid passing non-ASCII characters if possible.
        """

    def redirect(url):
        """
        Utility function that does a redirect.

        The request should have finish() called after this.
        """

    def setLastModified(when):
        """
        Set the C{Last-Modified} time for the response to this request.

        If I am called more than once, I ignore attempts to set Last-Modified
        earlier, only replacing the Last-Modified time if it is to a later
        value.

        If I am a conditional request, I may modify my response code to
        L{NOT_MODIFIED<http.NOT_MODIFIED>} if appropriate for the time given.

        @param when: The last time the resource being returned was modified, in
            seconds since the epoch.
        @type when: L{int} or L{float}

        @return: If I am a C{If-Modified-Since} conditional request and the time
            given is not newer than the condition, I return
            L{CACHED<http.CACHED>} to indicate that you should write no body.
            Otherwise, I return a false value.
        """

    def setETag(etag):
        """
        Set an C{entity tag} for the outgoing response.

        That's "entity tag" as in the HTTP/1.1 I{ETag} header, "used for
        comparing two or more entities from the same requested resource."

        If I am a conditional request, I may modify my response code to
        L{NOT_MODIFIED<http.NOT_MODIFIED>} or
        L{PRECONDITION_FAILED<http.PRECONDITION_FAILED>}, if appropriate for the
        tag given.

        @param etag: The entity tag for the resource being returned.
        @type etag: L{str}

        @return: If I am a C{If-None-Match} conditional request and the tag
            matches one in the request, I return L{CACHED<http.CACHED>} to
            indicate that you should write no body.  Otherwise, I return a
            false value.
        """

    def setHost(host, port, ssl=0):
        """
        Change the host and port the request thinks it's using.

        This method is useful for working with reverse HTTP proxies (e.g.  both
        Squid and Apache's mod_proxy can do this), when the address the HTTP
        client is using is different than the one we're listening on.

        For example, Apache may be listening on https://www.example.com, and
        then forwarding requests to http://localhost:8080, but we don't want
        HTML produced by Twisted to say 'http://localhost:8080', they should
        say 'https://www.example.com', so we do::

           request.setHost('www.example.com', 443, ssl=1)
        """