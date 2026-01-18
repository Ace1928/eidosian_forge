from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import properties
def _GetZonesInRegionLister(flag_names, region, compute_client, project):
    """Lists all the zones in a given region."""

    def Lister(*unused_args):
        """Returns a list of the zones for a given region."""
        if region:
            filter_expr = 'name eq {0}.*'.format(region)
        else:
            filter_expr = None
        errors = []
        global_resources = lister.GetGlobalResources(service=compute_client.apitools_client.zones, project=project, filter_expr=filter_expr, http=compute_client.apitools_client.http, batch_url=compute_client.batch_url, errors=errors)
        choices = [resource for resource in global_resources]
        if errors or not choices:
            punctuation = ':' if errors else '.'
            utils.RaiseToolException(errors, 'Unable to fetch a list of zones. Specifying [{0}] may fix this issue{1}'.format(', or '.join(flag_names), punctuation))
        return {compute_scope.ScopeEnum.ZONE: choices}
    return Lister