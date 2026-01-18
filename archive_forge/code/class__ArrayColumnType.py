from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from collections import OrderedDict
import re
from apitools.base.py import extra_types
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import zip
class _ArrayColumnType(_ColumnType):

    def __init__(self, scalar_type):
        super(_ArrayColumnType, self).__init__(scalar_type)

    def __eq__(self, other):
        return self.scalar_type == other.scalar_type and isinstance(other, _ArrayColumnType)

    def GetJsonValue(self, values):
        return extra_types.JsonValue(array_value=extra_types.JsonArray(entries=[ConvertJsonValueForScalarTypes(self.scalar_type, v) for v in values]))