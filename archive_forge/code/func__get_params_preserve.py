import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def _get_params_preserve(self, failobj, header):
    missing = object()
    value = self.get(header, missing)
    if value is missing:
        return failobj
    params = []
    for p in _parseparam(value):
        try:
            name, val = p.split('=', 1)
            name = name.strip()
            val = val.strip()
        except ValueError:
            name = p.strip()
            val = ''
        params.append((name, val))
    params = utils.decode_params(params)
    return params