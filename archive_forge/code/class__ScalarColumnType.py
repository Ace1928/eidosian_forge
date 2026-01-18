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
class _ScalarColumnType(_ColumnType):

    def __init__(self, scalar_type):
        super(_ScalarColumnType, self).__init__(scalar_type)

    def __eq__(self, other):
        return self.scalar_type == other.scalar_type and isinstance(other, _ScalarColumnType)

    def GetJsonValue(self, value):
        return ConvertJsonValueForScalarTypes(self.scalar_type, value)