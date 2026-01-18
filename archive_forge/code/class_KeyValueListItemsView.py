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
class KeyValueListItemsView(ListItemsView):

    def __iter__(self):
        for key, item in super(KeyValueListItemsView, self).__iter__():
            yield (key, getattr(item, self._mapping._value_field))