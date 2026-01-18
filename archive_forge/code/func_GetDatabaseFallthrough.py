from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def GetDatabaseFallthrough():
    """Python hook to get the value for the default database.

  Firestore currently only supports one database called '(default)'.

  Returns:
    The name of the default database.
  """
    return '(default)'