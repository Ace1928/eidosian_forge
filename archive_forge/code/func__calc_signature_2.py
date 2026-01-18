import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
def _calc_signature_2(self, params, verb, server_string, path):
    """Generate AWS signature version 2 string."""
    string_to_sign = '%s\n%s\n%s\n' % (verb, server_string, path)
    if self.hmac_256:
        current_hmac = self.hmac_256
        params['SignatureMethod'] = 'HmacSHA256'
    else:
        current_hmac = self.hmac
        params['SignatureMethod'] = 'HmacSHA1'
    string_to_sign += self._canonical_qs(params)
    current_hmac.update(string_to_sign.encode('utf-8'))
    b64 = base64.b64encode(current_hmac.digest()).decode('utf-8')
    return b64