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
def _DeletePrompt(kind, items):
    """Generates a delete prompt for a resource."""
    title = 'The following {} will be deleted.'
    if kind == constants.AWS_CLUSTER_KIND or kind == constants.AZURE_CLUSTER_KIND or kind == constants.ATTACHED_CLUSTER_KIND:
        title = title.format('clusters')
    elif kind == constants.AWS_NODEPOOL_KIND or kind == constants.AZURE_NODEPOOL_KIND:
        title = title.format('node pool')
    elif kind == constants.AZURE_CLIENT_KIND:
        title = title.format('client')
    console_io.PromptContinue(message=gke_util.ConstructList(title, items), throw_if_unattended=True, cancel_on_no=True)