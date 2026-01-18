from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
def _VisitInternal(self, value):
    ValidateType(value, list)
    result = []
    for item in value:
        result.append(self.element.ConvertValue(item))
    return result