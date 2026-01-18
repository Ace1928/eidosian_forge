import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
def _calc_signature_4(self, params, verb, server_string, path, headers, body_hash):
    """Generate AWS signature version 4 string."""

    def sign(key, msg):
        return hmac.new(key, self._get_utf8_value(msg), hashlib.sha256).digest()

    def signature_key(datestamp, region_name, service_name):
        """Signature key derivation.

            See http://docs.aws.amazon.com/general/latest/gr/
            signature-v4-examples.html#signature-v4-examples-python
            """
        k_date = sign(self._get_utf8_value(b'AWS4' + self.secret_key), datestamp)
        k_region = sign(k_date, region_name)
        k_service = sign(k_region, service_name)
        k_signing = sign(k_service, 'aws4_request')
        return k_signing

    def auth_param(param_name):
        """Get specified auth parameter.

            Provided via one of:
            - the Authorization header
            - the X-Amz-* query parameters
            """
        try:
            auth_str = headers['Authorization']
            param_str = auth_str.partition('%s=' % param_name)[2].split(',')[0]
        except KeyError:
            param_str = params.get('X-Amz-%s' % param_name)
        return param_str

    def date_param():
        """Get the X-Amz-Date' value.

            The value can be either a header or parameter.

            Note AWS supports parsing the Date header also, but this is not
            currently supported here as it will require some format mangling
            So the X-Amz-Date value must be YYYYMMDDTHHMMSSZ format, then it
            can be used to match against the YYYYMMDD format provided in the
            credential scope.
            see:
            http://docs.aws.amazon.com/general/latest/gr/
            sigv4-date-handling.html
            """
        try:
            return headers['X-Amz-Date']
        except KeyError:
            return params.get('X-Amz-Date')

    def canonical_header_str():
        headers_lower = dict(((k.lower().strip(), v.strip()) for k, v in headers.items()))
        user_agent = headers_lower.get('user-agent', '')
        strip_port = re.match('Boto/2\\.[0-9]\\.[0-2]', user_agent)
        header_list = []
        sh_str = auth_param('SignedHeaders')
        for h in sh_str.split(';'):
            if h not in headers_lower:
                continue
            if h == 'host' and strip_port:
                header_list.append('%s:%s' % (h, headers_lower[h].split(':')[0]))
                continue
            header_list.append('%s:%s' % (h, headers_lower[h]))
        return '\n'.join(header_list) + '\n'

    def canonical_query_str(verb, params):
        canonical_qs = ''
        if verb.upper() != 'POST':
            canonical_qs = self._canonical_qs(params)
        return canonical_qs
    cr = '\n'.join((verb.upper(), path, canonical_query_str(verb, params), canonical_header_str(), auth_param('SignedHeaders'), body_hash))
    credential = auth_param('Credential')
    credential_split = credential.split('/')
    credential_scope = '/'.join(credential_split[1:])
    credential_date = credential_split[1]
    param_date = date_param()
    if not param_date.startswith(credential_date):
        raise Exception(_('Request date mismatch error'))
    cr = cr.encode('utf-8')
    string_to_sign = '\n'.join(('AWS4-HMAC-SHA256', param_date, credential_scope, hashlib.sha256(cr).hexdigest()))
    req_region, req_service = credential_split[2:4]
    s_key = signature_key(credential_date, req_region, req_service)
    signature = hmac.new(s_key, self._get_utf8_value(string_to_sign), hashlib.sha256).hexdigest()
    return signature