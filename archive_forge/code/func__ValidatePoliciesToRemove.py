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
def _ValidatePoliciesToRemove(existing_cluster, updated_settings, policies_to_remove):
    """Checks if the policies specified for removal actually exist and that they are not updated in the same call.

  Args:
    existing_cluster: cluster before the update
    updated_settings: updated autoscale settings
    policies_to_remove: list of policy names to remove

  Raises:
    InvalidAutoscalingSettingsProvidedError: if the validation fails.
  """
    if not policies_to_remove:
        return
    if updated_settings and updated_settings.autoscaling_policies:
        for name in updated_settings.autoscaling_policies:
            if name in policies_to_remove:
                raise util.InvalidAutoscalingSettingsProvidedError(f"policy '{name}' specified both for update and removal")
    if not existing_cluster.autoscalingSettings:
        raise util.InvalidAutoscalingSettingsProvidedError(f"nonexistent policies '{policies_to_remove}' specified for removal")
    existing_policies = {p.key for p in existing_cluster.autoscalingSettings.autoscalingPolicies.additionalProperties}
    for name in policies_to_remove:
        if name not in existing_policies:
            raise util.InvalidAutoscalingSettingsProvidedError(f"nonexistent policies '{policies_to_remove}' specified for removal")