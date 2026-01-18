from __future__ import absolute_import, division, print_function
from passlib.utils.compat import PY3
import base64
import calendar
import json
import logging; log = logging.getLogger(__name__)
import math
import struct
import sys
import time as _time
import re
from warnings import warn
from passlib import exc
from passlib.exc import TokenError, MalformedTokenError, InvalidTokenError, UsedTokenError
from passlib.utils import (to_unicode, to_bytes, consteq,
from passlib.utils.binary import BASE64_CHARS, b32encode, b32decode
from passlib.utils.compat import (u, unicode, native_string_types, bascii_to_str, int_types, num_types,
from passlib.utils.decor import hybrid_method, memoized_property
from passlib.crypto.digest import lookup_hash, compile_hmac, pbkdf2_hmac
from passlib.hash import pbkdf2_sha256
@classmethod
def _from_parsed_uri(cls, result):
    """
        internal from_uri() helper --
        handles parsing a validated TOTP URI

        :param result:
            a urlparse() instance

        :returns:
            cls instance
        """
    label = result.path
    if label.startswith('/') and len(label) > 1:
        label = unquote(label[1:])
    else:
        raise cls._uri_parse_error('missing label')
    if ':' in label:
        try:
            issuer, label = label.split(':')
        except ValueError:
            raise cls._uri_parse_error('malformed label')
    else:
        issuer = None
    if label:
        label = label.strip() or None
    params = dict(label=label)
    for k, v in parse_qsl(result.query):
        if k in params:
            raise cls._uri_parse_error('duplicate parameter (%r)' % k)
        params[k] = v
    if issuer:
        if 'issuer' not in params:
            params['issuer'] = issuer
        elif params['issuer'] != issuer:
            raise cls._uri_parse_error('conflicting issuer identifiers')
    return cls(**cls._adapt_uri_params(**params))