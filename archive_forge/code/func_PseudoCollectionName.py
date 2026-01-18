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
def PseudoCollectionName(name):
    """Returns the pseudo collection name for name.

  Pseudo collection completion entities have no resource parser and/or URI.

  Args:
    name: The pseudo collection entity name.

  Returns:
    The pseudo collection name for name.
  """
    return '.'.join([_PSEUDO_COLLECTION_PREFIX, name])