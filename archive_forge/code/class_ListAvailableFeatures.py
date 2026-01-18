from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.ssl_policies import ssl_policies_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
class ListAvailableFeatures(base.ListCommand):
    """List available features that can be specified in an SSL policy.

  *{command}* lists available features that can be specified as part of the
  list of custom features in an SSL policy.

  An SSL policy specifies the server-side support for SSL features. An SSL
  policy can be attached to a TargetHttpsProxy or a TargetSslProxy. This affects
  connections between clients and the load balancer. SSL
  policies do not affect the connection between the load balancers and the
  backends. SSL policies are used by Application Load Balancers and proxy
  Network Load Balancers.
  """

    @classmethod
    def Args(cls, parser):
        """Set up arguments for this command."""
        parser.add_argument('--region', help='If provided, only features for the given region are shown.')
        parser.display_info.AddFormat('table([])')

    def Run(self, args):
        """Issues the request to list available SSL policy features."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        helper = ssl_policies_utils.SslPolicyHelper(holder)
        project = properties.VALUES.core.project.GetOrFail()
        return helper.ListAvailableFeatures(project, args.region if args.IsSpecified('region') else None)