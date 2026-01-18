import base64
import hashlib
import hmac
import logging
import random
import string
import sys
import traceback
import zlib
from saml2 import VERSION
from saml2 import saml
from saml2 import samlp
from saml2.time_util import instant
def _attrval(val, typ=''):
    if isinstance(val, list) or isinstance(val, set):
        attrval = [saml.AttributeValue(text=v) for v in val]
    elif val is None:
        attrval = None
    else:
        attrval = [saml.AttributeValue(text=val)]
    if typ:
        for ava in attrval:
            ava.set_type(typ)
    return attrval