from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.container.gkemulticloud import operations as op_api_util
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def Rollback(resource_ref=None, resource_client=None, args=None, kind=None, message=None):
    """Runs a rollback command for gkemulticloud.

  Args:
    resource_ref: obj, resource reference.
    resource_client: obj, client for the resource.
    args: obj, arguments parsed from the command.
    kind: str, the kind of resource e.g. AWS Cluster, Azure Node Pool.
    message: str, message to display while waiting for LRO to complete.

  Returns:
    The details of the updated resource.
  """
    _RollbackPrompt([message])
    async_ = getattr(args, 'async_', False)
    op = resource_client.Rollback(resource_ref, args)
    _LogAndWaitForOperation(op, async_, 'Rolling back ' + message)
    log.UpdatedResource(resource_ref, kind=kind, is_async=async_)
    return resource_client.Get(resource_ref)