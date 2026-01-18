from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ApplySubsettingArgs(client, args, backend_service, use_subset_size):
    """Applies the Subsetting argument(s) to the specified backend service.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.
    backend_service: The backend service object.
    use_subset_size: Should Subsetting.subset_size be used?
  """
    subsetting_args = {}
    add_subsetting = HasSubsettingArgs(args)
    if add_subsetting:
        subsetting_args['policy'] = client.messages.Subsetting.PolicyValueValuesEnum(args.subsetting_policy)
        if use_subset_size and HasSubsettingSubsetSizeArgs(args):
            subsetting_args['subsetSize'] = args.subsetting_subset_size
    if subsetting_args:
        backend_service.subsetting = client.messages.Subsetting(**subsetting_args)