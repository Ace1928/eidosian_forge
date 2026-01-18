from Markus Kuhn's C code, retrieved from:
from __future__ import division
import os
import sys
import warnings
from .table_wide import WIDE_EASTASIAN
from .table_zero import ZERO_WIDTH
from .unicode_versions import list_versions
@lru_cache(maxsize=8)
def _wcmatch_version(given_version):
    """
    Return nearest matching supported Unicode version level.

    If an exact match is not determined, the nearest lowest version level is
    returned after a warning is emitted.  For example, given supported levels
    ``4.1.0`` and ``5.0.0``, and a version string of ``4.9.9``, then ``4.1.0``
    is selected and returned:

    >>> _wcmatch_version('4.9.9')
    '4.1.0'
    >>> _wcmatch_version('8.0')
    '8.0.0'
    >>> _wcmatch_version('1')
    '4.1.0'

    :param str given_version: given version for compare, may be ``auto``
        (default), to select Unicode Version from Environment Variable,
        ``UNICODE_VERSION``. If the environment variable is not set, then the
        latest is used.
    :rtype: str
    :returns: unicode string, or non-unicode ``str`` type for python 2
        when given ``version`` is also type ``str``.
    """
    _return_str = not _PY3 and isinstance(given_version, str)
    if _return_str:
        unicode_versions = [ucs.encode() for ucs in list_versions()]
    else:
        unicode_versions = list_versions()
    latest_version = unicode_versions[-1]
    if given_version in (u'auto', 'auto'):
        given_version = os.environ.get('UNICODE_VERSION', 'latest' if not _return_str else latest_version.encode())
    if given_version in (u'latest', 'latest'):
        return latest_version if not _return_str else latest_version.encode()
    if given_version in unicode_versions:
        return given_version if not _return_str else given_version.encode()
    try:
        cmp_given = _wcversion_value(given_version)
    except ValueError:
        warnings.warn("UNICODE_VERSION value, {given_version!r}, is invalid. Value should be in form of `integer[.]+', the latest supported unicode version {latest_version!r} has been inferred.".format(given_version=given_version, latest_version=latest_version))
        return latest_version if not _return_str else latest_version.encode()
    earliest_version = unicode_versions[0]
    cmp_earliest_version = _wcversion_value(earliest_version)
    if cmp_given <= cmp_earliest_version:
        warnings.warn('UNICODE_VERSION value, {given_version!r}, is lower than any available unicode version. Returning lowest version level, {earliest_version!r}'.format(given_version=given_version, earliest_version=earliest_version))
        return earliest_version if not _return_str else earliest_version.encode()
    for idx, unicode_version in enumerate(unicode_versions):
        try:
            cmp_next_version = _wcversion_value(unicode_versions[idx + 1])
        except IndexError:
            return latest_version if not _return_str else latest_version.encode()
        if cmp_given == cmp_next_version[:len(cmp_given)]:
            return unicode_versions[idx + 1]
        if cmp_next_version > cmp_given:
            return unicode_version
    assert False, ('Code path unreachable', given_version, unicode_versions)