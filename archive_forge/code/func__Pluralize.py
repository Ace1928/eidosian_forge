from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _Pluralize(self, value):
    if not self.plural:
        if isinstance(value, list):
            return value[0] if value else None
        return value
    if value and (not isinstance(value, list)):
        return [value]
    return value if value else []