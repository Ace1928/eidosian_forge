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
class ResourceParamCompleter(ListCommandCompleter):
    """A completer that produces a resource list for one resource param."""

    def __init__(self, collection=None, param=None, **kwargs):
        super(ResourceParamCompleter, self).__init__(collection=collection, param=param, **kwargs)

    def RowToString(self, row, parameter_info=None):
        """Returns the string representation of row."""
        return row[self.column]