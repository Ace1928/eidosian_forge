import sys, datetime
from urllib.parse import urljoin, quote
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError as urllib_HTTPError
from .extras.httpheader import content_type, parse_http_datetime
from .host import preferred_suffixes
class URIOpener:
    """A wrapper around the urllib2 method to open a resource. Beyond accessing the data itself, the class
    sets a number of instance variable that might be relevant for processing.
    The class also adds an accept header to the outgoing request, namely
    text/html and application/xhtml+xml (unless set explicitly by the caller).
    
    If the content type is set by the server, the relevant HTTP response field is used. Otherwise,
    common suffixes are used (see L{host.preferred_suffixes}) to set the content type (this is really of importance
    for C{file:///} URI-s). If none of these works, the content type is empty.
        
    Interpretation of the content type for the return is done by Deron Meranda's U{httpheader module<http://deron.meranda.us/>}.
    
    @ivar data: the real data, ie, a file-like object
    @ivar headers: the return headers as sent back by the server
    @ivar content_type: the content type of the resource or the empty string, if the content type cannot be determined
    @ivar location: the real location of the data (ie, after possible redirection and content negotiation)
    @ivar last_modified_date: sets the last modified date if set in the header, None otherwise
    @ivar expiration_date: sets the expiration date if set in the header, I{current UTC plus one day} otherwise (this is used for caching purposes, hence this artificial setting)
    """
    CONTENT_LOCATION = 'Content-Location'
    CONTENT_TYPE = 'Content-Type'
    LAST_MODIFIED = 'Last-Modified'
    EXPIRES = 'Expires'

    def __init__(self, name, additional_headers={}, verify=True):
        """
        @param name: URL to be opened
        @keyword additional_headers: additional HTTP request headers to be added to the call
        """
        try:
            url = name.split('#')[0]
            if 'Accept' not in additional_headers:
                additional_headers['Accept'] = 'text/html, application/xhtml+xml'
            import requests
            r = requests.get(url, headers=additional_headers, verify=verify)
            self.data = r.content
            self.headers = r.headers
            if URIOpener.CONTENT_TYPE in self.headers:
                ct = content_type(self.headers[URIOpener.CONTENT_TYPE])
                self.content_type = ct.media_type
                if 'charset' in ct.parmdict:
                    self.charset = ct.parmdict['charset']
                else:
                    self.charset = None
            else:
                self.charset = None
                self.content_type = ''
                for suffix in preferred_suffixes.keys():
                    if name.endswith(suffix):
                        self.content_type = preferred_suffixes[suffix]
                        break
            if URIOpener.CONTENT_LOCATION in self.headers:
                self.location = urljoin(r.url, self.headers[URIOpener.CONTENT_LOCATION])
            else:
                self.location = name
            self.expiration_date = datetime.datetime.utcnow() + datetime.timedelta(days=1)
            if URIOpener.EXPIRES in self.headers:
                try:
                    self.expiration_date = parse_http_datetime(self.headers[URIOpener.EXPIRES])
                except:
                    pass
            self.last_modified_date = None
            if URIOpener.LAST_MODIFIED in self.headers:
                try:
                    self.last_modified_date = parse_http_datetime(self.headers[URIOpener.LAST_MODIFIED])
                except:
                    pass
        except urllib_HTTPError:
            e = sys.exc_info()[1]
            from . import HTTPError
            msg = BaseHTTPRequestHandler.responses[e.code]
            raise HTTPError('%s' % msg[1], e.code)
        except Exception:
            e = sys.exc_info()[1]
            from . import RDFaError
            raise RDFaError('%s' % e)