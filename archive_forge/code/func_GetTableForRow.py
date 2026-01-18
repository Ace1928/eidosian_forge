from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def GetTableForRow(self, row, parameter_info=None, create=True):
    """Returns the table for row.

    Args:
      row: The fully populated resource row.
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.
      create: Create the table if it doesn't exist if True.

    Returns:
      The table for row.
    """
    parameters = self._GetRuntimeParameters(parameter_info)
    values = [row[p.column] for p in parameters if p.aggregator]
    return self.cache.Table(self._GetTableName(suffix_list=values), columns=self.columns, keys=self.columns, timeout=self.timeout, create=create)