from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
def _ValidateMatch(regex, value, message):
    """Validate value matches regex."""
    matcher = regex.match(value)
    if not matcher:
        raise validation.ValidationError(message)
    return matcher