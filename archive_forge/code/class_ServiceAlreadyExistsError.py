from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
class ServiceAlreadyExistsError(exceptions.Error):
    """Error thrown if the service already exists and overwrite denied.
  """