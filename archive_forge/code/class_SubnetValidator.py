from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
class SubnetValidator(validation.Validator):
    """Checks that a subnet can be parsed and is a valid IPv4 or IPv6 subnet."""

    def Validate(self, value, unused_key=None):
        """Validates a subnet."""
        if value is None:
            raise validation.MissingAttribute('subnet must be specified')
        if not isinstance(value, six_subset.string_types):
            raise validation.ValidationError("subnet must be a string, not '%r'" % type(value))
        if ipaddr:
            try:
                ipaddr.IPNetwork(value)
            except ValueError:
                raise validation.ValidationError('%s is not a valid IPv4 or IPv6 subnet' % value)
        parts = value.split('/')
        if len(parts) == 2 and (not re.match('^[0-9]+$', parts[1])):
            raise validation.ValidationError('Prefix length of subnet %s must be an integer (quad-dotted masks are not supported)' % value)
        return value