from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class DmV2(base.Group):
    """Manage deployments of cloud resources.

  The {command} command group lets you manage the deployment of Google Cloud
  Platform resources using Google Cloud Deployment Manager.

  Google Cloud Deployment Manager allows you to specify all the resources needed
  for your application in a declarative format using YAML. You can also use
  Python or Jinja2 templates to parameterize the configuration and allow reuse
  of common deployment paradigms such as a load balanced, auto-scaled instance
  group.

  More information on Cloud Deployment Manager can be found here:
  https://cloud.google.com/deployment-manager and detailed documentation can be
  found here: https://cloud.google.com/deployment-manager/docs/
  """
    category = base.MANAGEMENT_TOOLS_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()