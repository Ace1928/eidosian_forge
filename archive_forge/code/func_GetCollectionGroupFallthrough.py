from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def GetCollectionGroupFallthrough():
    """Python hook to get the value for the '-' collection group.

  See details at:

  https://cloud.google.com/apis/design/design_patterns#list_sub-collections

  This allows us to describe or delete an index by specifying just its ID,
  without needing to know which collection group it belongs to.

  Returns:
    The value of the wildcard collection group.
  """
    return '-'