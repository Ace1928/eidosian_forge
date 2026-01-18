from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
def _PerformConversion(self, result):
    """Transforms the result value if a converter is specified."""
    return self.converter(result) if self.converter else result