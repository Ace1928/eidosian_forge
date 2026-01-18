from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import command
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def _get_hub_config(self, spec: messages.Message) -> messages.Message:
    if spec.policyControllerHubConfig is None:
        return self.messages.PolicyControllerHubConfig()
    return spec.policyControllerHubConfig