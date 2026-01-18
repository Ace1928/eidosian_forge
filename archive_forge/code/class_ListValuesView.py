from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
class ListValuesView(collections_abc.ValuesView):

    def __contains__(self, value):
        for v in iter(self):
            if v == value:
                return True
        return False

    def __iter__(self):
        for _, value in self._mapping.items():
            yield value