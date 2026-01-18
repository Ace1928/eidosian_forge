from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ApiGateway(base.Group):
    """Manage Cloud API Gateway resources.

  Commands for managing Cloud API Gateway resources.
  """
    category = base.API_PLATFORM_AND_ECOSYSTEMS_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()
        resources.REGISTRY.RegisterApiByName('apigateway', 'v1')