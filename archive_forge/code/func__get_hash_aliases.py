from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
def _get_hash_aliases(name):
    """
    internal helper used by :func:`lookup_hash` --
    normalize arbitrary hash name to hashlib format.
    if name not recognized, returns dummy record and issues a warning.

    :arg name:
        unnormalized name

    :returns:
        tuple with 2+ elements: ``(hashlib_name, iana_name|None, ... 0+ aliases)``.
    """
    orig = name
    if not isinstance(name, str):
        name = to_native_str(name, 'utf-8', 'hash name')
    name = re.sub('[_ /]', '-', name.strip().lower())
    if name.startswith('scram-'):
        name = name[6:]
        if name.endswith('-plus'):
            name = name[:-5]

    def check_table(name):
        for row in _known_hash_names:
            if name in row:
                return row
    result = check_table(name)
    if result:
        return result
    m = re.match('(?i)^(?P<name>[a-z]+)-?(?P<rev>\\d)?-?(?P<size>\\d{3,4})?$', name)
    if m:
        iana_name, rev, size = m.group('name', 'rev', 'size')
        if rev:
            iana_name += rev
        hashlib_name = iana_name
        if size:
            iana_name += '-' + size
            if rev:
                hashlib_name += '_'
            hashlib_name += size
        result = check_table(iana_name)
        if result:
            return result
        log.info('normalizing unrecognized hash name %r => %r / %r', orig, hashlib_name, iana_name)
    else:
        iana_name = name
        hashlib_name = name.replace('-', '_')
        log.warning('normalizing unrecognized hash name and format %r => %r / %r', orig, hashlib_name, iana_name)
    return (hashlib_name, iana_name)