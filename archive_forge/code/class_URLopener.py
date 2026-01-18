import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
class URLopener:
    """Class to open URLs.
    This is a class rather than just a subroutine because we may need
    more than one set of global protocol-specific options.
    Note -- this is a base class for those who don't want the
    automatic handling of errors type 302 (relocated) and 401
    (authorization needed)."""
    __tempfiles = None
    version = 'Python-urllib/%s' % __version__

    def __init__(self, proxies=None, **x509):
        msg = '%(class)s style of invoking requests is deprecated. Use newer urlopen functions/methods' % {'class': self.__class__.__name__}
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        if proxies is None:
            proxies = getproxies()
        assert hasattr(proxies, 'keys'), 'proxies must be a mapping'
        self.proxies = proxies
        self.key_file = x509.get('key_file')
        self.cert_file = x509.get('cert_file')
        self.addheaders = [('User-Agent', self.version), ('Accept', '*/*')]
        self.__tempfiles = []
        self.__unlink = os.unlink
        self.tempcache = None
        self.ftpcache = ftpcache

    def __del__(self):
        self.close()

    def close(self):
        self.cleanup()

    def cleanup(self):
        if self.__tempfiles:
            for file in self.__tempfiles:
                try:
                    self.__unlink(file)
                except OSError:
                    pass
            del self.__tempfiles[:]
        if self.tempcache:
            self.tempcache.clear()

    def addheader(self, *args):
        """Add a header to be used by the HTTP interface only
        e.g. u.addheader('Accept', 'sound/basic')"""
        self.addheaders.append(args)

    def open(self, fullurl, data=None):
        """Use URLopener().open(file) instead of open(file, 'r')."""
        fullurl = unwrap(_to_bytes(fullurl))
        fullurl = quote(fullurl, safe="%/:=&?~#+!$,;'@()*[]|")
        if self.tempcache and fullurl in self.tempcache:
            filename, headers = self.tempcache[fullurl]
            fp = open(filename, 'rb')
            return addinfourl(fp, headers, fullurl)
        urltype, url = _splittype(fullurl)
        if not urltype:
            urltype = 'file'
        if urltype in self.proxies:
            proxy = self.proxies[urltype]
            urltype, proxyhost = _splittype(proxy)
            host, selector = _splithost(proxyhost)
            url = (host, fullurl)
        else:
            proxy = None
        name = 'open_' + urltype
        self.type = urltype
        name = name.replace('-', '_')
        if not hasattr(self, name) or name == 'open_local_file':
            if proxy:
                return self.open_unknown_proxy(proxy, fullurl, data)
            else:
                return self.open_unknown(fullurl, data)
        try:
            if data is None:
                return getattr(self, name)(url)
            else:
                return getattr(self, name)(url, data)
        except (HTTPError, URLError):
            raise
        except OSError as msg:
            raise OSError('socket error', msg) from msg

    def open_unknown(self, fullurl, data=None):
        """Overridable interface to open unknown URL type."""
        type, url = _splittype(fullurl)
        raise OSError('url error', 'unknown url type', type)

    def open_unknown_proxy(self, proxy, fullurl, data=None):
        """Overridable interface to open unknown URL type."""
        type, url = _splittype(fullurl)
        raise OSError('url error', 'invalid proxy for %s' % type, proxy)

    def retrieve(self, url, filename=None, reporthook=None, data=None):
        """retrieve(url) returns (filename, headers) for a local object
        or (tempfilename, headers) for a remote object."""
        url = unwrap(_to_bytes(url))
        if self.tempcache and url in self.tempcache:
            return self.tempcache[url]
        type, url1 = _splittype(url)
        if filename is None and (not type or type == 'file'):
            try:
                fp = self.open_local_file(url1)
                hdrs = fp.info()
                fp.close()
                return (url2pathname(_splithost(url1)[1]), hdrs)
            except OSError:
                pass
        fp = self.open(url, data)
        try:
            headers = fp.info()
            if filename:
                tfp = open(filename, 'wb')
            else:
                garbage, path = _splittype(url)
                garbage, path = _splithost(path or '')
                path, garbage = _splitquery(path or '')
                path, garbage = _splitattr(path or '')
                suffix = os.path.splitext(path)[1]
                fd, filename = tempfile.mkstemp(suffix)
                self.__tempfiles.append(filename)
                tfp = os.fdopen(fd, 'wb')
            try:
                result = (filename, headers)
                if self.tempcache is not None:
                    self.tempcache[url] = result
                bs = 1024 * 8
                size = -1
                read = 0
                blocknum = 0
                if 'content-length' in headers:
                    size = int(headers['Content-Length'])
                if reporthook:
                    reporthook(blocknum, bs, size)
                while 1:
                    block = fp.read(bs)
                    if not block:
                        break
                    read += len(block)
                    tfp.write(block)
                    blocknum += 1
                    if reporthook:
                        reporthook(blocknum, bs, size)
            finally:
                tfp.close()
        finally:
            fp.close()
        if size >= 0 and read < size:
            raise ContentTooShortError('retrieval incomplete: got only %i out of %i bytes' % (read, size), result)
        return result

    def _open_generic_http(self, connection_factory, url, data):
        """Make an HTTP connection using connection_class.

        This is an internal method that should be called from
        open_http() or open_https().

        Arguments:
        - connection_factory should take a host name and return an
          HTTPConnection instance.
        - url is the url to retrieval or a host, relative-path pair.
        - data is payload for a POST request or None.
        """
        user_passwd = None
        proxy_passwd = None
        if isinstance(url, str):
            host, selector = _splithost(url)
            if host:
                user_passwd, host = _splituser(host)
                host = unquote(host)
            realhost = host
        else:
            host, selector = url
            proxy_passwd, host = _splituser(host)
            urltype, rest = _splittype(selector)
            url = rest
            user_passwd = None
            if urltype.lower() != 'http':
                realhost = None
            else:
                realhost, rest = _splithost(rest)
                if realhost:
                    user_passwd, realhost = _splituser(realhost)
                if user_passwd:
                    selector = '%s://%s%s' % (urltype, realhost, rest)
                if proxy_bypass(realhost):
                    host = realhost
        if not host:
            raise OSError('http error', 'no host given')
        if proxy_passwd:
            proxy_passwd = unquote(proxy_passwd)
            proxy_auth = base64.b64encode(proxy_passwd.encode()).decode('ascii')
        else:
            proxy_auth = None
        if user_passwd:
            user_passwd = unquote(user_passwd)
            auth = base64.b64encode(user_passwd.encode()).decode('ascii')
        else:
            auth = None
        http_conn = connection_factory(host)
        headers = {}
        if proxy_auth:
            headers['Proxy-Authorization'] = 'Basic %s' % proxy_auth
        if auth:
            headers['Authorization'] = 'Basic %s' % auth
        if realhost:
            headers['Host'] = realhost
        headers['Connection'] = 'close'
        for header, value in self.addheaders:
            headers[header] = value
        if data is not None:
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
            http_conn.request('POST', selector, data, headers)
        else:
            http_conn.request('GET', selector, headers=headers)
        try:
            response = http_conn.getresponse()
        except http.client.BadStatusLine:
            raise URLError('http protocol error: bad status line')
        if 200 <= response.status < 300:
            return addinfourl(response, response.msg, 'http:' + url, response.status)
        else:
            return self.http_error(url, response.fp, response.status, response.reason, response.msg, data)

    def open_http(self, url, data=None):
        """Use HTTP protocol."""
        return self._open_generic_http(http.client.HTTPConnection, url, data)

    def http_error(self, url, fp, errcode, errmsg, headers, data=None):
        """Handle http errors.

        Derived class can override this, or provide specific handlers
        named http_error_DDD where DDD is the 3-digit error code."""
        name = 'http_error_%d' % errcode
        if hasattr(self, name):
            method = getattr(self, name)
            if data is None:
                result = method(url, fp, errcode, errmsg, headers)
            else:
                result = method(url, fp, errcode, errmsg, headers, data)
            if result:
                return result
        return self.http_error_default(url, fp, errcode, errmsg, headers)

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        """Default error handler: close the connection and raise OSError."""
        fp.close()
        raise HTTPError(url, errcode, errmsg, headers, None)
    if _have_ssl:

        def _https_connection(self, host):
            return http.client.HTTPSConnection(host, key_file=self.key_file, cert_file=self.cert_file)

        def open_https(self, url, data=None):
            """Use HTTPS protocol."""
            return self._open_generic_http(self._https_connection, url, data)

    def open_file(self, url):
        """Use local file or FTP depending on form of URL."""
        if not isinstance(url, str):
            raise URLError('file error: proxy support for file protocol currently not implemented')
        if url[:2] == '//' and url[2:3] != '/' and (url[2:12].lower() != 'localhost/'):
            raise ValueError('file:// scheme is supported only on localhost')
        else:
            return self.open_local_file(url)

    def open_local_file(self, url):
        """Use local file."""
        import email.utils
        import mimetypes
        host, file = _splithost(url)
        localname = url2pathname(file)
        try:
            stats = os.stat(localname)
        except OSError as e:
            raise URLError(e.strerror, e.filename)
        size = stats.st_size
        modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
        mtype = mimetypes.guess_type(url)[0]
        headers = email.message_from_string('Content-Type: %s\nContent-Length: %d\nLast-modified: %s\n' % (mtype or 'text/plain', size, modified))
        if not host:
            urlfile = file
            if file[:1] == '/':
                urlfile = 'file://' + file
            return addinfourl(open(localname, 'rb'), headers, urlfile)
        host, port = _splitport(host)
        if not port and socket.gethostbyname(host) in (localhost(),) + thishost():
            urlfile = file
            if file[:1] == '/':
                urlfile = 'file://' + file
            elif file[:2] == './':
                raise ValueError('local file url may start with / or file:. Unknown url of type: %s' % url)
            return addinfourl(open(localname, 'rb'), headers, urlfile)
        raise URLError('local file error: not on local host')

    def open_ftp(self, url):
        """Use FTP protocol."""
        if not isinstance(url, str):
            raise URLError('ftp error: proxy support for ftp protocol currently not implemented')
        import mimetypes
        host, path = _splithost(url)
        if not host:
            raise URLError('ftp error: no host given')
        host, port = _splitport(host)
        user, host = _splituser(host)
        if user:
            user, passwd = _splitpasswd(user)
        else:
            passwd = None
        host = unquote(host)
        user = unquote(user or '')
        passwd = unquote(passwd or '')
        host = socket.gethostbyname(host)
        if not port:
            import ftplib
            port = ftplib.FTP_PORT
        else:
            port = int(port)
        path, attrs = _splitattr(path)
        path = unquote(path)
        dirs = path.split('/')
        dirs, file = (dirs[:-1], dirs[-1])
        if dirs and (not dirs[0]):
            dirs = dirs[1:]
        if dirs and (not dirs[0]):
            dirs[0] = '/'
        key = (user, host, port, '/'.join(dirs))
        if len(self.ftpcache) > MAXFTPCACHE:
            for k in list(self.ftpcache):
                if k != key:
                    v = self.ftpcache[k]
                    del self.ftpcache[k]
                    v.close()
        try:
            if key not in self.ftpcache:
                self.ftpcache[key] = ftpwrapper(user, passwd, host, port, dirs)
            if not file:
                type = 'D'
            else:
                type = 'I'
            for attr in attrs:
                attr, value = _splitvalue(attr)
                if attr.lower() == 'type' and value in ('a', 'A', 'i', 'I', 'd', 'D'):
                    type = value.upper()
            fp, retrlen = self.ftpcache[key].retrfile(file, type)
            mtype = mimetypes.guess_type('ftp:' + url)[0]
            headers = ''
            if mtype:
                headers += 'Content-Type: %s\n' % mtype
            if retrlen is not None and retrlen >= 0:
                headers += 'Content-Length: %d\n' % retrlen
            headers = email.message_from_string(headers)
            return addinfourl(fp, headers, 'ftp:' + url)
        except ftperrors() as exp:
            raise URLError(f'ftp error: {exp}') from exp

    def open_data(self, url, data=None):
        """Use "data" URL."""
        if not isinstance(url, str):
            raise URLError('data error: proxy support for data protocol currently not implemented')
        try:
            [type, data] = url.split(',', 1)
        except ValueError:
            raise OSError('data error', 'bad data URL')
        if not type:
            type = 'text/plain;charset=US-ASCII'
        semi = type.rfind(';')
        if semi >= 0 and '=' not in type[semi:]:
            encoding = type[semi + 1:]
            type = type[:semi]
        else:
            encoding = ''
        msg = []
        msg.append('Date: %s' % time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(time.time())))
        msg.append('Content-type: %s' % type)
        if encoding == 'base64':
            data = base64.decodebytes(data.encode('ascii')).decode('latin-1')
        else:
            data = unquote(data)
        msg.append('Content-Length: %d' % len(data))
        msg.append('')
        msg.append(data)
        msg = '\n'.join(msg)
        headers = email.message_from_string(msg)
        f = io.StringIO(msg)
        return addinfourl(f, headers, url)