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
def _GetTableName(self, suffix_list=None):
    """Returns the table name; the module path if no collection.

    Args:
      suffix_list: a list of values to attach to the end of the table name.
        Typically, these will be aggregator values, like project ID.
    Returns: a name to use for the table in the cache DB.
    """
    if self.collection:
        name = [self.collection]
    else:
        name = [module_util.GetModulePath(self)]
    if suffix_list:
        name.extend(suffix_list)
    return '.'.join(name)