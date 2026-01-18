from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
def ValidateNotType(source_value, non_expected_type):
    if isinstance(source_value, non_expected_type):
        raise ValueError('Did not expect %s for value %s' % (non_expected_type, source_value))