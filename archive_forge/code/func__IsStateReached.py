from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute.instance_groups.managed import wait_info
from googlecloudsdk.core import log
from googlecloudsdk.core.util import retry
def _IsStateReached(client, group_ref, desired_igm_state):
    """Checks if the desired state is reached."""
    responses, errors = _GetResources(client, group_ref)
    if errors:
        utils.RaiseToolException(errors)
    if desired_igm_state == IgmState.STABLE:
        is_stable = responses[0].status.isStable
        if not is_stable:
            log.out.Print(wait_info.CreateWaitText(responses[0]))
        return is_stable
    elif desired_igm_state == IgmState.VERSION_TARGET_REACHED:
        is_version_target_reached = responses[0].status.versionTarget.isReached
        if not is_version_target_reached:
            log.out.Print('Waiting for group to reach version target')
        return is_version_target_reached
    elif desired_igm_state == IgmState.ALL_INSTANCES_CONFIG_EFFECTIVE:
        all_instances_config_effective = responses[0].status.allInstancesConfig.effective
        if not all_instances_config_effective:
            log.out.Print('Waiting for group to reach all-instances config effective')
        return all_instances_config_effective
    else:
        raise Exception('Incorrect desired_igm_state')