import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
def _calc_signature_0(self, params):
    """Generate AWS signature version 0 string."""
    s = (params['Action'] + params['Timestamp']).encode('utf-8')
    self.hmac.update(s)
    return base64.b64encode(self.hmac.digest()).decode('utf-8')