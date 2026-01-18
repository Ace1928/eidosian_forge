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
def DeleteWithIgnoreErrors(args, resource_client, resource_ref, message, kind):
    """Calls the delete command and suggests using --ignore-errors on failure.

  Args:
    args: obj, arguments parsed from the command.
    resource_client: obj, client for the resource.
    resource_ref: obj, resource reference.
    message: str, message to display while waiting for LRO to complete.
    kind: str, the kind of resource e.g. AWS Cluster, Azure Node Pool.

  Returns:
    The details of the updated resource.
  """
    res = 'cluster'
    if kind == constants.AWS_NODEPOOL_KIND or kind == constants.AZURE_NODEPOOL_KIND:
        res = 'node pool'
    if not args.ignore_errors:
        _PromptIgnoreErrors(args, resource_client, resource_ref)
    try:
        ret = Delete(resource_ref=resource_ref, resource_client=resource_client, args=args, message=message, kind=kind)
    except waiter.OperationError as e:
        if not args.ignore_errors:
            pretty_print.Info('Delete {} failed. Try re-running with `--ignore-errors`.\n'.format(res))
        raise e
    except apitools_exceptions.HttpError as e:
        if not args.ignore_errors:
            pretty_print.Info('Delete {} failed. Try re-running with `--ignore-errors`.\n'.format(res))
        raise e
    return ret