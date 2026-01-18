from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.api_lib.util import resource_search
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.cache import completion_cache
from googlecloudsdk.core.cache import resource_cache
import six
def _GRI_RowToString(self, row, parameter_info=None):
    parts = list(row)
    for column, parameter in enumerate(self.parameters):
        if parameter.name in self.qualified_parameter_names:
            continue
        value = parameter_info.GetValue(parameter.name)
        if parts[column] != value:
            break
        parts[column] = ''
    if 'collection' in self.qualified_parameter_names:
        collection = self.collection
        is_fully_qualified = True
    else:
        collection = None
        is_fully_qualified = True
    return six.text_type(resources.GRI(reversed(parts), collection=collection, is_fully_qualified=is_fully_qualified))