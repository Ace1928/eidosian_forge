import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
class HmacAuthV4Handler(AuthHandler, HmacKeys):
    """
    Implements the new Version 4 HMAC authorization.
    """
    capability = ['hmac-v4']

    def __init__(self, host, config, provider, service_name=None, region_name=None):
        AuthHandler.__init__(self, host, config, provider)
        HmacKeys.__init__(self, host, config, provider)
        self.service_name = service_name
        self.region_name = region_name

    def _sign(self, key, msg, hex=False):
        if not isinstance(key, bytes):
            key = key.encode('utf-8')
        if hex:
            sig = hmac.new(key, msg.encode('utf-8'), sha256).hexdigest()
        else:
            sig = hmac.new(key, msg.encode('utf-8'), sha256).digest()
        return sig

    def headers_to_sign(self, http_request):
        """
        Select the headers from the request that need to be included
        in the StringToSign.
        """
        host_header_value = self.host_header(self.host, http_request)
        if http_request.headers.get('Host'):
            host_header_value = http_request.headers['Host']
        headers_to_sign = {'Host': host_header_value}
        for name, value in http_request.headers.items():
            lname = name.lower()
            if lname.startswith('x-amz'):
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                headers_to_sign[name] = value
        return headers_to_sign

    def host_header(self, host, http_request):
        port = http_request.port
        secure = http_request.protocol == 'https'
        if port == 80 and (not secure) or (port == 443 and secure):
            return host
        return '%s:%s' % (host, port)

    def query_string(self, http_request):
        parameter_names = sorted(http_request.params.keys())
        pairs = []
        for pname in parameter_names:
            pval = get_utf8able_str(http_request.params[pname])
            pairs.append(urllib.parse.quote(pname, safe=''.encode('ascii')) + '=' + urllib.parse.quote(pval, safe='-_~'.encode('ascii')))
        return '&'.join(pairs)

    def canonical_query_string(self, http_request):
        if http_request.method == 'POST':
            return ''
        l = []
        for param in sorted(http_request.params):
            value = get_utf8able_str(http_request.params[param])
            l.append('%s=%s' % (urllib.parse.quote(param, safe='-_.~'), urllib.parse.quote(value, safe='-_.~')))
        return '&'.join(l)

    def canonical_headers(self, headers_to_sign):
        """
        Return the headers that need to be included in the StringToSign
        in their canonical form by converting all header keys to lower
        case, sorting them in alphabetical order and then joining
        them into a string, separated by newlines.
        """
        canonical = []
        normalized_headers = {}
        for header in headers_to_sign:
            c_name = header.lower().strip()
            raw_value = str(headers_to_sign[header])
            if '"' in raw_value:
                c_value = raw_value.strip()
            else:
                c_value = ' '.join(raw_value.strip().split())
            normalized_headers[c_name] = c_value
        for key in sorted(normalized_headers):
            canonical.append('%s:%s' % (key, normalized_headers[key]))
        return '\n'.join(canonical)

    def signed_headers(self, headers_to_sign):
        l = ['%s' % n.lower().strip() for n in headers_to_sign]
        l = sorted(l)
        return ';'.join(l)

    def canonical_uri(self, http_request):
        path = http_request.auth_path
        normalized = posixpath.normpath(path).replace('\\', '/')
        encoded = urllib.parse.quote(normalized)
        if len(path) > 1 and path.endswith('/'):
            encoded += '/'
        return encoded

    def payload(self, http_request):
        body = http_request.body
        if hasattr(body, 'seek') and hasattr(body, 'read'):
            return boto.utils.compute_hash(body, hash_algorithm=sha256)[0]
        elif not isinstance(body, bytes):
            body = body.encode('utf-8')
        return sha256(body).hexdigest()

    def canonical_request(self, http_request):
        cr = [http_request.method.upper()]
        cr.append(self.canonical_uri(http_request))
        cr.append(self.canonical_query_string(http_request))
        headers_to_sign = self.headers_to_sign(http_request)
        cr.append(self.canonical_headers(headers_to_sign) + '\n')
        cr.append(self.signed_headers(headers_to_sign))
        cr.append(self.payload(http_request))
        return '\n'.join(cr)

    def scope(self, http_request):
        scope = [self._provider.access_key]
        scope.append(http_request.timestamp)
        scope.append(http_request.region_name)
        scope.append(http_request.service_name)
        scope.append('aws4_request')
        return '/'.join(scope)

    def split_host_parts(self, host):
        return host.split('.')

    def determine_region_name(self, host):
        parts = self.split_host_parts(host)
        if self.region_name is not None:
            region_name = self.region_name
        elif len(parts) > 1:
            if parts[1] == 'us-gov':
                region_name = 'us-gov-west-1'
            elif len(parts) == 3:
                region_name = 'us-east-1'
            else:
                region_name = parts[1]
        else:
            region_name = parts[0]
        return region_name

    def determine_service_name(self, host):
        parts = self.split_host_parts(host)
        if self.service_name is not None:
            service_name = self.service_name
        else:
            service_name = parts[0]
        return service_name

    def credential_scope(self, http_request):
        scope = []
        http_request.timestamp = http_request.headers['X-Amz-Date'][0:8]
        scope.append(http_request.timestamp)
        region_name = self.determine_region_name(http_request.host)
        service_name = self.determine_service_name(http_request.host)
        http_request.service_name = service_name
        http_request.region_name = region_name
        scope.append(http_request.region_name)
        scope.append(http_request.service_name)
        scope.append('aws4_request')
        return '/'.join(scope)

    def string_to_sign(self, http_request, canonical_request):
        """
        Return the canonical StringToSign as well as a dict
        containing the original version of all headers that
        were included in the StringToSign.
        """
        sts = ['AWS4-HMAC-SHA256']
        sts.append(http_request.headers['X-Amz-Date'])
        sts.append(self.credential_scope(http_request))
        sts.append(sha256(canonical_request.encode('utf-8')).hexdigest())
        return '\n'.join(sts)

    def signature(self, http_request, string_to_sign):
        key = self._provider.secret_key
        k_date = self._sign(('AWS4' + key).encode('utf-8'), http_request.timestamp)
        k_region = self._sign(k_date, http_request.region_name)
        k_service = self._sign(k_region, http_request.service_name)
        k_signing = self._sign(k_service, 'aws4_request')
        return self._sign(k_signing, string_to_sign, hex=True)

    def add_auth(self, req, **kwargs):
        """
        Add AWS4 authentication to a request.

        :type req: :class`boto.connection.HTTPRequest`
        :param req: The HTTPRequest object.
        """
        if 'X-Amzn-Authorization' in req.headers:
            del req.headers['X-Amzn-Authorization']
        now = datetime.datetime.utcnow()
        req.headers['X-Amz-Date'] = now.strftime('%Y%m%dT%H%M%SZ')
        if self._provider.security_token:
            req.headers['X-Amz-Security-Token'] = self._provider.security_token
        qs = self.query_string(req)
        qs_to_post = qs
        if 'unmangled_req' in kwargs:
            qs_to_post = self.query_string(kwargs['unmangled_req'])
        if qs_to_post and req.method == 'POST':
            req.body = qs_to_post
            req.headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
            req.headers['Content-Length'] = str(len(req.body))
        else:
            req.path = req.path.split('?')[0]
            if qs:
                req.path = req.path + '?' + qs
        canonical_request = self.canonical_request(req)
        boto.log.debug('CanonicalRequest:\n%s' % canonical_request)
        string_to_sign = self.string_to_sign(req, canonical_request)
        boto.log.debug('StringToSign:\n%s' % string_to_sign)
        signature = self.signature(req, string_to_sign)
        boto.log.debug('Signature:\n%s' % signature)
        headers_to_sign = self.headers_to_sign(req)
        l = ['AWS4-HMAC-SHA256 Credential=%s' % self.scope(req)]
        l.append('SignedHeaders=%s' % self.signed_headers(headers_to_sign))
        l.append('Signature=%s' % signature)
        req.headers['Authorization'] = ','.join(l)