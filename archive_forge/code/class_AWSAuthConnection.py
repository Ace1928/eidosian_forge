from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
class AWSAuthConnection(object):

    def __init__(self, host, aws_access_key_id=None, aws_secret_access_key=None, is_secure=True, port=None, proxy=None, proxy_port=None, proxy_user=None, proxy_pass=None, debug=0, https_connection_factory=None, path='/', provider='aws', security_token=None, suppress_consec_slashes=True, validate_certs=True, profile_name=None):
        """
        :type host: str
        :param host: The host to make the connection to

        :keyword str aws_access_key_id: Your AWS Access Key ID (provided by
            Amazon). If none is specified, the value in your
            ``AWS_ACCESS_KEY_ID`` environmental variable is used.
        :keyword str aws_secret_access_key: Your AWS Secret Access Key
            (provided by Amazon). If none is specified, the value in your
            ``AWS_SECRET_ACCESS_KEY`` environmental variable is used.
        :keyword str security_token: The security token associated with
            temporary credentials issued by STS.  Optional unless using
            temporary credentials.  If none is specified, the environment
            variable ``AWS_SECURITY_TOKEN`` is used if defined.

        :type is_secure: boolean
        :param is_secure: Whether the connection is over SSL

        :type https_connection_factory: list or tuple
        :param https_connection_factory: A pair of an HTTP connection
            factory and the exceptions to catch.  The factory should have
            a similar interface to L{http_client.HTTPSConnection}.

        :param str proxy: Address/hostname for a proxy server

        :type proxy_port: int
        :param proxy_port: The port to use when connecting over a proxy

        :type proxy_user: str
        :param proxy_user: The username to connect with on the proxy

        :type proxy_pass: str
        :param proxy_pass: The password to use when connection over a proxy.

        :type port: int
        :param port: The port to use to connect

        :type suppress_consec_slashes: bool
        :param suppress_consec_slashes: If provided, controls whether
            consecutive slashes will be suppressed in key paths.

        :type validate_certs: bool
        :param validate_certs: Controls whether SSL certificates
            will be validated or not.  Defaults to True.

        :type profile_name: str
        :param profile_name: Override usual Credentials section in config
            file to use a named set of keys instead.
        """
        self.suppress_consec_slashes = suppress_consec_slashes
        self.num_retries = 6
        if config.has_option('Boto', 'is_secure'):
            is_secure = config.getboolean('Boto', 'is_secure')
        self.is_secure = is_secure
        self.https_validate_certificates = config.getbool('Boto', 'https_validate_certificates', validate_certs)
        if self.https_validate_certificates and (not HAVE_HTTPS_CONNECTION):
            raise BotoClientError('SSL server certificate validation is enabled in boto configuration, but Python dependencies required to support this feature are not available. Certificate validation is only supported when running under Python 2.6 or later.')
        certs_file = config.get_value('Boto', 'ca_certificates_file', DEFAULT_CA_CERTS_FILE)
        if certs_file == 'system':
            certs_file = None
        self.ca_certificates_file = certs_file
        if port:
            self.port = port
        else:
            self.port = PORTS_BY_SECURITY[is_secure]
        self.handle_proxy(proxy, proxy_port, proxy_user, proxy_pass)
        self.http_exceptions = (http_client.HTTPException, socket.error, socket.gaierror, http_client.BadStatusLine)
        self.http_unretryable_exceptions = []
        if HAVE_HTTPS_CONNECTION:
            self.http_unretryable_exceptions.append(https_connection.InvalidCertificateException)
        self.socket_exception_values = (errno.EINTR,)
        if https_connection_factory is not None:
            self.https_connection_factory = https_connection_factory[0]
            self.http_exceptions += https_connection_factory[1]
        else:
            self.https_connection_factory = None
        if is_secure:
            self.protocol = 'https'
        else:
            self.protocol = 'http'
        self.host = host
        self.path = path
        if not isinstance(debug, six.integer_types):
            debug = 0
        self.debug = config.getint('Boto', 'debug', debug)
        self.host_header = None
        self.http_connection_kwargs = {}
        if (sys.version_info[0], sys.version_info[1]) >= (2, 6):
            self.http_connection_kwargs['timeout'] = config.getint('Boto', 'http_socket_timeout', 70)
        is_anonymous_connection = getattr(self, 'anon', False)
        if isinstance(provider, Provider):
            self.provider = provider
        else:
            self._provider_type = provider
            self.provider = Provider(self._provider_type, aws_access_key_id, aws_secret_access_key, security_token, profile_name, anon=is_anonymous_connection)
        if self.provider.host:
            self.host = self.provider.host
        if self.provider.port:
            self.port = self.provider.port
        if self.provider.host_header:
            self.host_header = self.provider.host_header
        self._pool = ConnectionPool()
        self._connection = (self.host, self.port, self.is_secure)
        self._last_rs = None
        self._auth_handler = auth.get_auth_handler(host, config, self.provider, self._required_auth_capability())
        if getattr(self, 'AuthServiceName', None) is not None:
            self.auth_service_name = self.AuthServiceName
        self.request_hook = None

    def __repr__(self):
        return '%s:%s' % (self.__class__.__name__, self.host)

    def _required_auth_capability(self):
        return []

    def _get_auth_service_name(self):
        return getattr(self._auth_handler, 'service_name')

    def _set_auth_service_name(self, value):
        self._auth_handler.service_name = value
    auth_service_name = property(_get_auth_service_name, _set_auth_service_name)

    def _get_auth_region_name(self):
        return getattr(self._auth_handler, 'region_name')

    def _set_auth_region_name(self, value):
        self._auth_handler.region_name = value
    auth_region_name = property(_get_auth_region_name, _set_auth_region_name)

    def connection(self):
        return self.get_http_connection(*self._connection)
    connection = property(connection)

    def aws_access_key_id(self):
        return self.provider.access_key
    aws_access_key_id = property(aws_access_key_id)
    gs_access_key_id = aws_access_key_id
    access_key = aws_access_key_id

    def aws_secret_access_key(self):
        return self.provider.secret_key
    aws_secret_access_key = property(aws_secret_access_key)
    gs_secret_access_key = aws_secret_access_key
    secret_key = aws_secret_access_key

    def profile_name(self):
        return self.provider.profile_name
    profile_name = property(profile_name)

    def get_path(self, path='/'):
        if not self.suppress_consec_slashes:
            return self.path + re.sub('^(/*)/', '\\1', path)
        pos = path.find('?')
        if pos >= 0:
            params = path[pos:]
            path = path[:pos]
        else:
            params = None
        if path[-1] == '/':
            need_trailing = True
        else:
            need_trailing = False
        path_elements = self.path.split('/')
        path_elements.extend(path.split('/'))
        path_elements = [p for p in path_elements if p]
        path = '/' + '/'.join(path_elements)
        if path[-1] != '/' and need_trailing:
            path += '/'
        if params:
            path = path + params
        return path

    def server_name(self, port=None):
        if not port:
            port = self.port
        if port == 80:
            signature_host = self.host
        else:
            ver_int = sys.version_info[0] * 10 + sys.version_info[1]
            if port == 443 and ver_int >= 26:
                signature_host = self.host
            else:
                signature_host = '%s:%d' % (self.host, port)
        return signature_host

    def handle_proxy(self, proxy, proxy_port, proxy_user, proxy_pass):
        self.proxy = proxy
        self.proxy_port = proxy_port
        self.proxy_user = proxy_user
        self.proxy_pass = proxy_pass
        if 'http_proxy' in os.environ and (not self.proxy):
            pattern = re.compile('(?:http://)?(?:(?P<user>[\\w\\-\\.]+):(?P<pass>.*)@)?(?P<host>[\\w\\-\\.]+)(?::(?P<port>\\d+))?')
            match = pattern.match(os.environ['http_proxy'])
            if match:
                self.proxy = match.group('host')
                self.proxy_port = match.group('port')
                self.proxy_user = match.group('user')
                self.proxy_pass = match.group('pass')
        else:
            if not self.proxy:
                self.proxy = config.get_value('Boto', 'proxy', None)
            if not self.proxy_port:
                self.proxy_port = config.get_value('Boto', 'proxy_port', None)
            if not self.proxy_user:
                self.proxy_user = config.get_value('Boto', 'proxy_user', None)
            if not self.proxy_pass:
                self.proxy_pass = config.get_value('Boto', 'proxy_pass', None)
        if not self.proxy_port and self.proxy:
            print('http_proxy environment variable does not specify a port, using default')
            self.proxy_port = self.port
        self.no_proxy = os.environ.get('no_proxy', '') or os.environ.get('NO_PROXY', '')
        self.use_proxy = self.proxy is not None

    def get_http_connection(self, host, port, is_secure):
        conn = self._pool.get_http_connection(host, port, is_secure)
        if conn is not None:
            return conn
        else:
            return self.new_http_connection(host, port, is_secure)

    def skip_proxy(self, host):
        if not self.no_proxy:
            return False
        if self.no_proxy == '*':
            return True
        hostonly = host
        hostonly = host.split(':')[0]
        for name in self.no_proxy.split(','):
            if name and (hostonly.endswith(name) or host.endswith(name)):
                return True
        return False

    def new_http_connection(self, host, port, is_secure):
        if host is None:
            host = self.server_name()
        host = boto.utils.parse_host(host)
        http_connection_kwargs = self.http_connection_kwargs.copy()
        http_connection_kwargs['port'] = port
        if self.use_proxy and (not is_secure) and (not self.skip_proxy(host)):
            host = self.proxy
            http_connection_kwargs['port'] = int(self.proxy_port)
        if is_secure:
            boto.log.debug('establishing HTTPS connection: host=%s, kwargs=%s', host, http_connection_kwargs)
            if self.use_proxy and (not self.skip_proxy(host)):
                connection = self.proxy_ssl(host, is_secure and 443 or 80)
            elif self.https_connection_factory:
                connection = self.https_connection_factory(host)
            elif self.https_validate_certificates and HAVE_HTTPS_CONNECTION:
                connection = https_connection.CertValidatingHTTPSConnection(host, ca_certs=self.ca_certificates_file, **http_connection_kwargs)
            else:
                connection = http_client.HTTPSConnection(host, **http_connection_kwargs)
        else:
            boto.log.debug('establishing HTTP connection: kwargs=%s' % http_connection_kwargs)
            if self.https_connection_factory:
                connection = self.https_connection_factory(host, **http_connection_kwargs)
            else:
                connection = http_client.HTTPConnection(host, **http_connection_kwargs)
        if self.debug > 1:
            connection.set_debuglevel(self.debug)
        if host.split(':')[0] == self.host and is_secure == self.is_secure:
            self._connection = (host, port, is_secure)
        connection.response_class = HTTPResponse
        return connection

    def put_http_connection(self, host, port, is_secure, connection):
        self._pool.put_http_connection(host, port, is_secure, connection)

    def proxy_ssl(self, host=None, port=None):
        if host and port:
            host = '%s:%d' % (host, port)
        else:
            host = '%s:%d' % (self.host, self.port)
        timeout = self.http_connection_kwargs.get('timeout')
        if timeout is not None:
            sock = socket.create_connection((self.proxy, int(self.proxy_port)), timeout)
        else:
            sock = socket.create_connection((self.proxy, int(self.proxy_port)))
        boto.log.debug('Proxy connection: CONNECT %s HTTP/1.0\r\n', host)
        sock.sendall(six.ensure_binary('CONNECT %s HTTP/1.0\r\n' % host))
        sock.sendall(six.ensure_binary('User-Agent: %s\r\n' % UserAgent))
        if self.proxy_user and self.proxy_pass:
            for k, v in self.get_proxy_auth_header().items():
                sock.sendall(six.ensure_binary('%s: %s\r\n' % (k, v)))
            if config.getbool('Boto', 'send_crlf_after_proxy_auth_headers', False):
                sock.sendall(six.ensure_binary('\r\n'))
        else:
            sock.sendall(six.ensure_binary('\r\n'))
        resp = http_client.HTTPResponse(sock, debuglevel=self.debug)
        resp.begin()
        if resp.status != 200:
            raise socket.error(-71, six.ensure_binary('Error talking to HTTP proxy %s:%s: %s (%s)' % (self.proxy, self.proxy_port, resp.status, resp.reason)))
        resp.close()
        h = http_client.HTTPConnection(host)
        if self.https_validate_certificates and HAVE_HTTPS_CONNECTION:
            msg = 'wrapping ssl socket for proxied connection; '
            if self.ca_certificates_file:
                msg += 'CA certificate file=%s' % self.ca_certificates_file
            else:
                msg += 'using system provided SSL certs'
            boto.log.debug(msg)
            key_file = self.http_connection_kwargs.get('key_file', None)
            cert_file = self.http_connection_kwargs.get('cert_file', None)
            context = ssl.create_default_context()
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True
            if cert_file:
                context.load_cert_chain(cert_file, key_file)
            context.load_verify_locations(self.ca_certificates_file)
            sslSock = context.wrap_socket(sock, server_hostname=self.host)
            cert = sslSock.getpeercert()
            hostname = self.host.split(':', 0)[0]
            if not https_connection.ValidateCertificateHostname(cert, hostname):
                raise https_connection.InvalidCertificateException(hostname, cert, 'hostname mismatch')
        elif hasattr(http_client, 'ssl'):
            sslSock = http_client.ssl.SSLSocket(sock)
        else:
            sslSock = socket.ssl(sock, None, None)
            sslSock = http_client.FakeSocket(sock, sslSock)
        h.sock = sslSock
        return h

    def prefix_proxy_to_path(self, path, host=None):
        path = self.protocol + '://' + (host or self.server_name()) + path
        return path

    def get_proxy_auth_header(self):
        auth = encodebytes(self.proxy_user + ':' + self.proxy_pass)
        return {'Proxy-Authorization': 'Basic %s' % auth}

    def get_proxy_url_with_auth(self):
        if not self.use_proxy:
            return None
        if self.proxy_user or self.proxy_pass:
            if self.proxy_pass:
                login_info = '%s:%s@' % (self.proxy_user, self.proxy_pass)
            else:
                login_info = '%s@' % self.proxy_user
        else:
            login_info = ''
        return 'http://%s%s:%s' % (login_info, self.proxy, str(self.proxy_port or self.port))

    def set_host_header(self, request):
        try:
            request.headers['Host'] = self._auth_handler.host_header(self.host, request)
        except AttributeError:
            request.headers['Host'] = self.host.split(':', 1)[0]

    def set_request_hook(self, hook):
        self.request_hook = hook

    def _mexe(self, request, sender=None, override_num_retries=None, retry_handler=None):
        """
        mexe - Multi-execute inside a loop, retrying multiple times to handle
               transient Internet errors by simply trying again.
               Also handles redirects.

        This code was inspired by the S3Utils classes posted to the boto-users
        Google group by Larry Bates.  Thanks!

        """
        boto.log.debug('Method: %s' % request.method)
        boto.log.debug('Path: %s' % request.path)
        boto.log.debug('Data: %s' % request.body)
        boto.log.debug('Headers: %s' % request.headers)
        boto.log.debug('Host: %s' % request.host)
        boto.log.debug('Port: %s' % request.port)
        boto.log.debug('Params: %s' % request.params)
        response = None
        body = None
        ex = None
        if override_num_retries is None:
            num_retries = config.getint('Boto', 'num_retries', self.num_retries)
        else:
            num_retries = override_num_retries
        i = 0
        connection = self.get_http_connection(request.host, request.port, self.is_secure)
        if not isinstance(request.body, bytes) and hasattr(request.body, 'encode'):
            request.body = request.body.encode('utf-8')
        while i <= num_retries:
            next_sleep = min(random.random() * 2 ** i, float(boto.config.get('Boto', 'max_retry_delay', 60)))
            try:
                boto.log.debug('Token: %s' % self.provider.security_token)
                request.authorize(connection=self)
                if 's3' not in self._required_auth_capability():
                    if not getattr(self, 'anon', False):
                        if not request.headers.get('Host'):
                            self.set_host_header(request)
                boto.log.debug('Final headers: %s' % request.headers)
                request.start_time = datetime.now()
                if callable(sender):
                    response = sender(connection, request.method, request.path, request.body, request.headers)
                else:
                    connection.request(request.method, request.path, request.body, request.headers)
                    response = connection.getresponse()
                boto.log.debug('Response headers: %s' % response.getheaders())
                location = response.getheader('location')
                if request.method == 'HEAD' and getattr(response, 'chunked', False):
                    response.chunked = 0
                if callable(retry_handler):
                    status = retry_handler(response, i, next_sleep)
                    if status:
                        msg, i, next_sleep = status
                        if msg:
                            boto.log.debug(msg)
                        time.sleep(next_sleep)
                        continue
                if response.status in [500, 502, 503, 504]:
                    msg = 'Received %d response.  ' % response.status
                    msg += 'Retrying in %3.1f seconds' % next_sleep
                    boto.log.debug(msg)
                    body = response.read()
                    if isinstance(body, bytes):
                        body = body.decode('utf-8')
                elif response.status < 300 or response.status >= 400 or (not location):
                    conn_header_value = response.getheader('connection')
                    if conn_header_value == 'close':
                        connection.close()
                    else:
                        self.put_http_connection(request.host, request.port, self.is_secure, connection)
                    if self.request_hook is not None:
                        self.request_hook.handle_request_data(request, response)
                    return response
                else:
                    scheme, request.host, request.path, params, query, fragment = urlparse(location)
                    if query:
                        request.path += '?' + query
                    if ':' in request.host:
                        request.host, request.port = request.host.split(':', 1)
                    msg = 'Redirecting: %s' % scheme + '://'
                    msg += request.host + request.path
                    boto.log.debug(msg)
                    connection = self.get_http_connection(request.host, request.port, scheme == 'https')
                    response = None
                    continue
            except PleaseRetryException as e:
                boto.log.debug('encountered a retry exception: %s' % e)
                connection = self.new_http_connection(request.host, request.port, self.is_secure)
                response = e.response
                ex = e
            except self.http_exceptions as e:
                for unretryable in self.http_unretryable_exceptions:
                    if isinstance(e, unretryable):
                        boto.log.debug('encountered unretryable %s exception, re-raising' % e.__class__.__name__)
                        raise
                boto.log.debug('encountered %s exception, reconnecting' % e.__class__.__name__)
                connection = self.new_http_connection(request.host, request.port, self.is_secure)
                ex = e
            time.sleep(next_sleep)
            i += 1
        if self.request_hook is not None:
            self.request_hook.handle_request_data(request, response, error=True)
        if response:
            raise BotoServerError(response.status, response.reason, body)
        elif ex:
            raise ex
        else:
            msg = 'Please report this exception as a Boto Issue!'
            raise BotoClientError(msg)

    def build_base_http_request(self, method, path, auth_path, params=None, headers=None, data='', host=None):
        path = self.get_path(path)
        if auth_path is not None:
            auth_path = self.get_path(auth_path)
        if params is None:
            params = {}
        else:
            params = params.copy()
        if headers is None:
            headers = {}
        else:
            headers = headers.copy()
        if self.host_header and (not boto.utils.find_matching_headers('host', headers)):
            headers['host'] = self.host_header
        host = host or self.host
        if self.use_proxy and (not self.skip_proxy(host)):
            if not auth_path:
                auth_path = path
            path = self.prefix_proxy_to_path(path, host)
            if self.proxy_user and self.proxy_pass and (not self.is_secure):
                headers.update(self.get_proxy_auth_header())
        return HTTPRequest(method, self.protocol, host, self.port, path, auth_path, params, headers, data)

    def _find_s3_host(self, endpoint):
        ix = endpoint.rfind('.s3.')
        if ix == -1:
            return None
        return ix + 1

    def _get_s3_host(self, endpoint):
        ix = self._find_s3_host(endpoint)
        if ix:
            return endpoint[ix:]

    def _change_s3_host(self, endpoint, new_host):
        ix = self._find_s3_host(endpoint)
        if ix:
            return endpoint[:ix] + new_host

    def _fix_s3_endpoint_region(self, endpoint, correct_region):
        """Return a new bucket endpoint that uses correct_region.
        Return None if host substitution is not possible.
        """
        if not (endpoint and correct_region):
            return None
        new_host = 's3.%s.amazonaws.com' % correct_region
        new_endpoint = self._change_s3_host(endpoint, new_host)
        if new_endpoint:
            return new_endpoint
        boto.log.debug('Could not change s3 host for %s' % endpoint)

    def _get_correct_s3_endpoint_from_response(self, request, err, get_header):
        """Attempt to return a new s3 endpoint using the correct region to
        access a bucket. Return None if a retry is not possible."""
        if callable(get_header):
            region = get_header('x-amz-bucket-region')
            if region:
                boto.log.debug('Got correct region from response headers.')
                return self._fix_s3_endpoint_region(request.host, region)
        if err.region:
            boto.log.debug('Got correct region from parsed xml in err.region.')
            return self._fix_s3_endpoint_region(request.host, err.region)
        elif err.error_code == 'IllegalLocationConstraintException':
            region_regex = 'The (.*?) location constraint is incompatible for the region specific endpoint this request was sent to\\.'
            match = re.search(region_regex, err.body)
            if match and match.group(1) != 'unspecified':
                region = match.group(1)
                boto.log.debug('Got correct region from response body.')
                return self._fix_s3_endpoint_region(request.host, region)
        elif err.endpoint:
            boto.log.debug('Got correct endpoint from response body.')
            return err.endpoint
        boto.log.debug('Sending a bucket HEAD request to get correct region.')
        req = self.build_base_http_request('HEAD', '/', '/', {}, None, '', request.host)
        bucket_head_response = self._mexe(req, None, None)
        region = bucket_head_response.getheader('x-amz-bucket-region')
        if region:
            boto.log.debug('Got correct region from a bucket head request.')
            return self._fix_s3_endpoint_region(request.host, region)

    def _change_s3_host_from_error(self, request, err, get_header=None):
        new_endpoint = self._get_correct_s3_endpoint_from_response(request, err, get_header)
        if not new_endpoint:
            return None
        msg = 'This request was sent to %s, ' % request.host
        msg += 'when it should have been sent to %s. ' % new_endpoint
        request.host = new_endpoint
        new_host = self._get_s3_host(new_endpoint)
        if new_host and new_host != self.host:
            msg += 'This error may have arisen because your S3 host, '
            msg += 'currently %s, is configured incorrectly. ' % self.host
            msg += 'Please change your configuration to use %s ' % new_host
            msg += 'to avoid multiple unnecessary redirects '
            msg += 'and signing attempts.'
            self.host = new_host
        boto.log.debug(msg)
        return request

    def _get_request_for_s3_retry(self, http_request, response, err):
        if response:
            body = response.read()
            if body:
                body = body.decode('utf-8')
            err = S3ResponseError(response.status, response.reason, body)
            return self._change_s3_host_from_error(http_request, err, get_header=response.getheader)
        elif err:
            return self._change_s3_host_from_error(http_request, err)

    def make_request(self, method, path, headers=None, data='', host=None, auth_path=None, sender=None, override_num_retries=None, params=None, retry_handler=None):
        """Make a request to the server.
        Include logic for retrying on s3 region errors.
        """
        if params is None:
            params = {}
        http_request = self.build_base_http_request(method, path, auth_path, params, headers, data, host)
        response, err = (None, None)
        try:
            response = self._mexe(http_request, sender, override_num_retries, retry_handler=retry_handler)
        except S3ResponseError as e:
            err = e
        status = (response or err).status
        if http_request.host.endswith('amazonaws.com') and status in [301, 400]:
            retry_request = self._get_request_for_s3_retry(http_request, response, err)
            if retry_request:
                return self._mexe(retry_request, sender, override_num_retries, retry_handler=retry_handler)
        if response:
            return response
        elif err:
            raise err

    def close(self):
        """(Optional) Close any open HTTP connections.  This is non-destructive,
        and making a new request will open a connection again."""
        boto.log.debug('closing all HTTP connections')
        self._connection = None