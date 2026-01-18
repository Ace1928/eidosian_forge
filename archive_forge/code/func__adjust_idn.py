from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
import sys
import traceback
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse, ParseResult
from ._asn1 import serialize_asn1_string_as_der
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
from .basic import (
from ._objects import (
from ._obj2txt import obj2txt
def _adjust_idn(value, idn_rewrite):
    if idn_rewrite == 'ignore' or not value:
        return value
    if idn_rewrite == 'idna' and _is_ascii(value):
        return value
    if idn_rewrite not in ('idna', 'unicode'):
        raise ValueError('Invalid value for idn_rewrite: "{0}"'.format(idn_rewrite))
    if not HAS_IDNA:
        raise OpenSSLObjectError(missing_required_lib('idna', reason='to transform {what} DNS name "{name}" to {dest}'.format(name=value, what='IDNA' if idn_rewrite == 'unicode' else 'Unicode', dest='Unicode' if idn_rewrite == 'unicode' else 'IDNA')))
    parts = value.split(u'.')
    for index, part in enumerate(parts):
        if part in (u'', u'*'):
            continue
        try:
            if idn_rewrite == 'idna':
                parts[index] = idna.encode(part).decode('ascii')
            elif idn_rewrite == 'unicode' and part.startswith(u'xn--'):
                parts[index] = idna.decode(part)
        except idna.IDNAError as exc2008:
            try:
                if idn_rewrite == 'idna':
                    parts[index] = part.encode('idna').decode('ascii')
                elif idn_rewrite == 'unicode' and part.startswith(u'xn--'):
                    parts[index] = part.encode('ascii').decode('idna')
            except Exception as exc2003:
                raise OpenSSLObjectError(u'Error while transforming part "{part}" of {what} DNS name "{name}" to {dest}. IDNA2008 transformation resulted in "{exc2008}", IDNA2003 transformation resulted in "{exc2003}".'.format(part=part, name=value, what='IDNA' if idn_rewrite == 'unicode' else 'Unicode', dest='Unicode' if idn_rewrite == 'unicode' else 'IDNA', exc2003=exc2003, exc2008=exc2008))
    return u'.'.join(parts)