from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
class DispatchEntryURLValidator(validation.Validator):
    """Validater for URL patterns."""

    def Validate(self, value, unused_key=None):
        """Validates an URL pattern."""
        if value is None:
            raise validation.MissingAttribute('url must be specified')
        if not isinstance(value, six_subset.string_types):
            raise validation.ValidationError("url must be a string, not '%r'" % type(value))
        url_holder = ParsedURL(value)
        if url_holder.host_exact:
            _ValidateMatch(_URL_HOST_EXACT_PATTERN_RE, url_holder.host, "invalid host_pattern '%s'" % url_holder.host)
            _ValidateNotIpV4Address(url_holder.host)
        else:
            _ValidateMatch(_URL_HOST_SUFFIX_PATTERN_RE, url_holder.host_pattern, "invalid host_pattern '%s'" % url_holder.host_pattern)
        return value