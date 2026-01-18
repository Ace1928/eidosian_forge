from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
class ValueValidator(validation.Validator):
    """Ensures that values in `HttpHeadersDict` are valid.

    An instance is used as `HttpHeadersDict`'s `VALUE_VALIDATOR`.
    """

    def Validate(self, value, key=None):
        """Returns a value, or raises an exception if the value is invalid.

      According to `RFC 2616 section 4.2`_ header field values must consist "of
      either *TEXT or combinations of token, separators, and quoted-string"::

          TEXT = <any OCTET except CTLs, but including LWS>

      Args:
        value: HTTP header field value.
        key: HTTP header field name.

      Returns:
        A value argument.

      Raises:
        appinfo_errors.InvalidHttpHeaderValue: An argument cannot be used as an
            HTTP header value.

      .. _RFC 2616, section 4.2:
         https://www.ietf.org/rfc/rfc2616.txt
      """
        error = appinfo_errors.InvalidHttpHeaderValue('HTTP header values must not contain non-ASCII data')
        if isinstance(value, six_subset.string_types):
            b_value = EnsureAsciiBytes(value, error)
        else:
            b_value = EnsureAsciiBytes('%s' % value, error)
        key = key.lower()
        printable = set(string.printable[:-5].encode('ascii'))
        if not all((b in printable for b in b_value)):
            raise appinfo_errors.InvalidHttpHeaderValue('HTTP header field values must consist of printable characters.')
        HttpHeadersDict.ValueValidator.AssertHeaderNotTooLong(key, value)
        return value

    @staticmethod
    def AssertHeaderNotTooLong(name, value):
        header_length = len(('%s: %s\r\n' % (name, value)).encode('ascii'))
        if header_length >= HttpHeadersDict.MAX_HEADER_LENGTH:
            try:
                max_len = HttpHeadersDict.MAX_HEADER_VALUE_LENGTHS[name]
            except KeyError:
                raise appinfo_errors.InvalidHttpHeaderValue('HTTP header (name + value) is too long.')
            if len(value) > max_len:
                insert = (name, len(value), max_len)
                raise appinfo_errors.InvalidHttpHeaderValue('%r header value has length %d, which exceed the maximum allowed, %d.' % insert)