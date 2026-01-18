from __future__ import unicode_literals
import codecs
from email import message_from_file
import json
import logging
import re
from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE
def _best_version(fields):
    """Detect the best version depending on the fields used."""

    def _has_marker(keys, markers):
        return any((marker in keys for marker in markers))
    keys = [key for key, value in fields.items() if value not in ([], 'UNKNOWN', None)]
    possible_versions = ['1.0', '1.1', '1.2', '1.3', '2.1', '2.2']
    for key in keys:
        if key not in _241_FIELDS and '1.0' in possible_versions:
            possible_versions.remove('1.0')
            logger.debug('Removed 1.0 due to %s', key)
        if key not in _314_FIELDS and '1.1' in possible_versions:
            possible_versions.remove('1.1')
            logger.debug('Removed 1.1 due to %s', key)
        if key not in _345_FIELDS and '1.2' in possible_versions:
            possible_versions.remove('1.2')
            logger.debug('Removed 1.2 due to %s', key)
        if key not in _566_FIELDS and '1.3' in possible_versions:
            possible_versions.remove('1.3')
            logger.debug('Removed 1.3 due to %s', key)
        if key not in _566_FIELDS and '2.1' in possible_versions:
            if key != 'Description':
                possible_versions.remove('2.1')
                logger.debug('Removed 2.1 due to %s', key)
        if key not in _643_FIELDS and '2.2' in possible_versions:
            possible_versions.remove('2.2')
            logger.debug('Removed 2.2 due to %s', key)
    if len(possible_versions) == 1:
        return possible_versions[0]
    elif len(possible_versions) == 0:
        logger.debug('Out of options - unknown metadata set: %s', fields)
        raise MetadataConflictError('Unknown metadata set')
    is_1_1 = '1.1' in possible_versions and _has_marker(keys, _314_MARKERS)
    is_1_2 = '1.2' in possible_versions and _has_marker(keys, _345_MARKERS)
    is_2_1 = '2.1' in possible_versions and _has_marker(keys, _566_MARKERS)
    is_2_2 = '2.2' in possible_versions and _has_marker(keys, _643_MARKERS)
    if int(is_1_1) + int(is_1_2) + int(is_2_1) + int(is_2_2) > 1:
        raise MetadataConflictError('You used incompatible 1.1/1.2/2.1/2.2 fields')
    if not is_1_1 and (not is_1_2) and (not is_2_1) and (not is_2_2):
        if PKG_INFO_PREFERRED_VERSION in possible_versions:
            return PKG_INFO_PREFERRED_VERSION
    if is_1_1:
        return '1.1'
    if is_1_2:
        return '1.2'
    if is_2_1:
        return '2.1'
    return '2.2'