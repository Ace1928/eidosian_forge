from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.vmware import clusters
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.vmware import flags
from googlecloudsdk.command_lib.vmware.clusters import util
from googlecloudsdk.core import log
def _RemoveAutoscalingPolicies(autoscaling_settings: util.AutoscalingSettings, policies_to_remove: List[str]) -> util.AutoscalingSettings:
    if not policies_to_remove:
        return autoscaling_settings
    for policy in policies_to_remove:
        del autoscaling_settings.autoscaling_policies[policy]
    return autoscaling_settings