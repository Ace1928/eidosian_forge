from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.core import log
class ConfigmanagementFeatureState(object):
    """Feature state class stores ACM status."""

    def __init__(self, clusterName):
        self.name = clusterName
        self.config_sync = NA
        self.last_synced_token = NA
        self.last_synced = NA
        self.sync_branch = NA
        self.policy_controller_state = NA
        self.hierarchy_controller_state = NA
        self.version = NA
        self.upgrades = NA

    def update_sync_state(self, fs):
        """Update config_sync state for the membership that has ACM installed.

    Args:
      fs: ConfigManagementFeatureState
    """
        if not (fs.configSyncState and fs.configSyncState.syncState):
            self.config_sync = 'SYNC_STATE_UNSPECIFIED'
        else:
            self.config_sync = fs.configSyncState.syncState.code
            if fs.configSyncState.syncState.syncToken:
                self.last_synced_token = fs.configSyncState.syncState.syncToken[:7]
            self.last_synced = fs.configSyncState.syncState.lastSyncTime
            if has_config_sync_git(fs):
                self.sync_branch = fs.membershipSpec.configSync.git.syncBranch

    def update_policy_controller_state(self, md):
        """Update policy controller state for the membership that has ACM installed.

    Args:
      md: MembershipFeatureState
    """
        if md.state.code.name != 'OK':
            self.policy_controller_state = 'ERROR: {}'.format(md.state.description)
            return
        fs = md.configmanagement
        if not (fs.policyControllerState and fs.policyControllerState.deploymentState):
            self.policy_controller_state = NA
            return
        pc_deployment_state = fs.policyControllerState.deploymentState
        expected_deploys = {'GatekeeperControllerManager': pc_deployment_state.gatekeeperControllerManagerState}
        if fs.membershipSpec and fs.membershipSpec.version and (fs.membershipSpec.version > '1.4.1'):
            expected_deploys['GatekeeperAudit'] = pc_deployment_state.gatekeeperAudit
        for deployment_name, deployment_state in expected_deploys.items():
            if not deployment_state:
                continue
            elif deployment_state.name != 'INSTALLED':
                self.policy_controller_state = '{} {}'.format(deployment_name, deployment_state)
                return
            self.policy_controller_state = deployment_state.name

    def update_hierarchy_controller_state(self, fs):
        """Update hierarchy controller state for the membership that has ACM installed.

    The PENDING state is set separately after this logic. The PENDING state
    suggests the HC part in feature_spec and feature_state are inconsistent, but
    the HC status from feature_state is not ERROR. This suggests that HC might
    be still in the updating process, so we mark it as PENDING

    Args:
      fs: ConfigmanagementFeatureState
    """
        if not (fs.hierarchyControllerState and fs.hierarchyControllerState.state):
            self.hierarchy_controller_state = NA
            return
        hc_deployment_state = fs.hierarchyControllerState.state
        hnc_state = 'NOT_INSTALLED'
        ext_state = 'NOT_INSTALLED'
        if hc_deployment_state.hnc:
            hnc_state = hc_deployment_state.hnc.name
        if hc_deployment_state.extension:
            ext_state = hc_deployment_state.extension.name
        deploys_to_status = {('INSTALLED', 'INSTALLED'): 'INSTALLED', ('INSTALLED', 'NOT_INSTALLED'): 'INSTALLED', ('NOT_INSTALLED', 'NOT_INSTALLED'): NA}
        if (hnc_state, ext_state) in deploys_to_status:
            self.hierarchy_controller_state = deploys_to_status[hnc_state, ext_state]
        else:
            self.hierarchy_controller_state = 'ERROR'

    def update_pending_state(self, feature_spec_mc, feature_state_mc):
        """Update config sync and policy controller with the pending state.

    Args:
      feature_spec_mc: MembershipConfig
      feature_state_mc: MembershipConfig
    """
        feature_state_pending = feature_state_mc is None and feature_spec_mc is not None
        if feature_state_pending:
            self.last_synced_token = 'PENDING'
            self.last_synced = 'PENDING'
            self.sync_branch = 'PENDING'
        if self.config_sync.__str__() in ['SYNCED', 'NOT_CONFIGURED', 'NOT_INSTALLED', NA] and (feature_state_pending or feature_spec_mc.configSync != feature_state_mc.configSync):
            self.config_sync = 'PENDING'
        if self.policy_controller_state.__str__() in ['INSTALLED', 'GatekeeperAudit NOT_INSTALLED', NA] and feature_state_pending:
            self.policy_controller_state = 'PENDING'
        if self.hierarchy_controller_state.__str__() != 'ERROR' and feature_state_pending or feature_spec_mc.hierarchyController != feature_state_mc.hierarchyController:
            self.hierarchy_controller_state = 'PENDING'