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
class KeyValidator(validation.Validator):
    """Ensures that keys in `HttpHeadersDict` are valid.

    `HttpHeadersDict` contains a list of headers. An instance is used as
    `HttpHeadersDict`'s `KEY_VALIDATOR`.
    """

    def Validate(self, name, unused_key=None):
        """Returns an argument, or raises an exception if the argument is invalid.

      HTTP header names are defined by `RFC 2616, section 4.2`_.

      Args:
        name: HTTP header field value.
        unused_key: Unused.

      Returns:
        name argument, unchanged.

      Raises:
        appinfo_errors.InvalidHttpHeaderName: An argument cannot be used as an
            HTTP header name.

      .. _RFC 2616, section 4.2:
         https://www.ietf.org/rfc/rfc2616.txt
      """
        original_name = name
        if isinstance(name, six_subset.string_types):
            name = EnsureAsciiBytes(name, appinfo_errors.InvalidHttpHeaderName('HTTP header values must not contain non-ASCII data'))
        name = name.lower().decode('ascii')
        if not _HTTP_TOKEN_RE.match(name):
            raise appinfo_errors.InvalidHttpHeaderName('An HTTP header must be a non-empty RFC 2616 token.')
        if name in _HTTP_REQUEST_HEADERS:
            raise appinfo_errors.InvalidHttpHeaderName('%r can only be used in HTTP requests, not responses.' % original_name)
        if name.startswith('x-appengine'):
            raise appinfo_errors.InvalidHttpHeaderName('HTTP header names that begin with X-Appengine are reserved.')
        if wsgiref.util.is_hop_by_hop(name):
            raise appinfo_errors.InvalidHttpHeaderName('Only use end-to-end headers may be used. See RFC 2616 section 13.5.1.')
        if name in HttpHeadersDict.DISALLOWED_HEADERS:
            raise appinfo_errors.InvalidHttpHeaderName('%s is a disallowed header.' % name)
        return original_name