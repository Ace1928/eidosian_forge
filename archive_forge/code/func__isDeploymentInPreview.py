from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.api_lib.deployment_manager import dm_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.deployment_manager import alpha_flags
def _isDeploymentInPreview(self, args):
    deployment = dm_api_util.FetchDeployment(self.client, self.messages, dm_base.GetProject(), args.deployment)
    if deployment.update:
        return True
    return False