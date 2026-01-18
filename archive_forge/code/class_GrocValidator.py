from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
import sys
import traceback
from googlecloudsdk.third_party.appengine._internal import six_subset
class GrocValidator(validation.Validator):
    """Checks that a schedule is in valid groc format."""

    def Validate(self, value, key=None):
        """Validates a schedule."""
        if value is None:
            raise validation.MissingAttribute('schedule must be specified')
        if not isinstance(value, six_subset.string_types):
            raise TypeError("schedule must be a string, not '%r'" % type(value))
        if groc and groctimespecification:
            try:
                groctimespecification.GrocTimeSpecification(value)
            except groc.GrocException as e:
                raise validation.ValidationError("schedule '%s' failed to parse: %s" % (value, e.args[0]))
        return value