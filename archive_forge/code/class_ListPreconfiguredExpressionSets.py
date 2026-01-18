from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
class ListPreconfiguredExpressionSets(base.ListCommand):
    """List all available preconfigured expression sets.

  *{command}* lists all available preconfigured expression sets that can be used
  with the Cloud Armor rules language.

  ## EXAMPLES

  To list all current preconfigured expressions sets run this:

    $ {command}
  """

    @staticmethod
    def Args(parser):
        """Set up arguments for this command."""
        base.URI_FLAG.RemoveFromParser(parser)
        parser.display_info.AddFormat('\n        table(id:label=EXPRESSION_SET,\n              aliases:format="get([])",\n              expressions:format="table(id:label=RULE_ID,\n                                        sensitivity:label=SENSITIVITY)")\n    ')

    def Run(self, args):
        """Issues the request to list available preconfigured expression sets."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client.apitools_client
        messages = client.MESSAGES_MODULE
        project = properties.VALUES.core.project.Get(required=True)
        request = messages.ComputeSecurityPoliciesListPreconfiguredExpressionSetsRequest(project=project)
        response = client.securityPolicies.ListPreconfiguredExpressionSets(request)
        if response.preconfiguredExpressionSets is not None:
            return response.preconfiguredExpressionSets.wafRules.expressionSets
        return response.preconfiguredExpressionSets