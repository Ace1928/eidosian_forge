from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources as fleet_resources
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
class RolloutFlagParser:
    """Parse flags during fleet rollout command runtime."""

    def __init__(self, args: parser_extensions.Namespace, release_track: base.ReleaseTrack):
        self.args = args
        self.release_track = release_track
        self.messages = util.GetMessagesModule(release_track)

    def IsEmpty(self, message: messages.Message) -> bool:
        """Determines if a message is empty.

    Args:
      message: A message to check the emptiness.

    Returns:
      A bool indictating if the message is equivalent to a newly initialized
      empty message instance.
    """
        return message == type(message)()

    def TrimEmpty(self, message: messages.Message):
        """Trim empty messages to avoid cluttered request."""
        if not self.IsEmpty(message):
            return message
        return None

    def Rollout(self) -> fleet_messages.Rollout:
        rollout = fleet_messages.Rollout()
        rollout.name = util.RolloutName(self.args)
        rollout.displayName = self._DisplayName()
        rollout.labels = self._Labels()
        rollout.managedRolloutConfig = self._ManagedRolloutConfig()
        rollout.feature = self._FeatureUpdate()
        return rollout

    def _DisplayName(self) -> str:
        return self.args.display_name

    def _Labels(self) -> fleet_messages.Rollout.LabelsValue:
        """Parses --labels."""
        if '--labels' not in self.args.GetSpecifiedArgs():
            return None
        labels = self.args.labels
        labels_value = fleet_messages.Rollout.LabelsValue()
        for key, value in labels.items():
            labels_value.additionalProperties.append(fleet_messages.Rollout.LabelsValue.AdditionalProperty(key=key, value=value))
        return labels_value

    def _ManagedRolloutConfig(self) -> fleet_messages.ManagedRolloutConfig:
        managed_rollout_config = fleet_messages.ManagedRolloutConfig()
        managed_rollout_config.soakDuration = self._SoakDuration()
        return self.TrimEmpty(managed_rollout_config)

    def _SoakDuration(self) -> str:
        """Parses --soak-duration.

    Accepts ISO 8601 durations format. To read more,
    https://cloud.google.com/sdk/gcloud/reference/topic/

    Returns:
      str, in standard duration format, in unit of seconds.
    """
        if '--soak-duration' not in self.args.GetSpecifiedArgs():
            return None
        return '{}s'.format(self.args.soak_duration)

    def _FeatureUpdate(self) -> fleet_messages.FeatureUpdate:
        """Constructs message FeatureUpdate."""
        feature_update = fleet_messages.FeatureUpdate()
        feature_update.securityPostureConfig = self._SecurityPostureConfig()
        feature_update.binaryAuthorizationConfig = self._BinaryAuthorzationConfig()
        return self.TrimEmpty(feature_update)

    def _SecurityPostureConfig(self) -> fleet_messages.SecurityPostureConfig:
        security_posture_config = fleet_messages.SecurityPostureConfig()
        security_posture_config.mode = self._SecurityPostureMode()
        security_posture_config.vulnerabilityMode = self._VulnerabilityModeValueValuesEnum()
        return self.TrimEmpty(security_posture_config)

    def _SecurityPostureMode(self) -> fleet_messages.SecurityPostureConfig.ModeValueValuesEnum:
        """Parses --security-posture."""
        if '--security-posture' not in self.args.GetSpecifiedArgs():
            return None
        enum_type = fleet_messages.SecurityPostureConfig.ModeValueValuesEnum
        mapping = {'disabled': enum_type.DISABLED, 'standard': enum_type.BASIC}
        return mapping[self.args.security_posture]

    def _VulnerabilityModeValueValuesEnum(self) -> fleet_messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum:
        """Parses --workload-vulnerability-scanning."""
        if '--workload-vulnerability-scanning' not in self.args.GetSpecifiedArgs():
            return None
        enum_type = self.messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum
        mapping = {'disabled': enum_type.VULNERABILITY_DISABLED, 'standard': enum_type.VULNERABILITY_BASIC, 'enterprise': enum_type.VULNERABILITY_ENTERPRISE}
        return mapping[self.args.workload_vulnerability_scanning]

    def _BinaryAuthorzationConfig(self) -> fleet_messages.BinaryAuthorizationConfig:
        binary_authorization_config = fleet_messages.BinaryAuthorizationConfig()
        binary_authorization_config.evaluationMode = self._EvaluationMode()
        binary_authorization_config.policyBindings = list(self._PolicyBindings())
        return self.TrimEmpty(binary_authorization_config)

    def _EvaluationMode(self) -> fleet_messages.BinaryAuthorizationConfig.EvaluationModeValueValuesEnum:
        """Parses --binauthz-evaluation-mode."""
        if '--binauthz-evaluation-mode' not in self.args.GetSpecifiedArgs():
            return None
        enum_type = self.messages.BinaryAuthorizationConfig.EvaluationModeValueValuesEnum
        mapping = {'disabled': enum_type.DISABLED, 'policy-bindings': enum_type.POLICY_BINDINGS}
        return mapping[self.args.binauthz_evaluation_mode]

    def _PolicyBindings(self) -> Iterator[fleet_messages.PolicyBinding]:
        """Parses --binauthz-policy-bindings."""
        if '--binauthz-policy-bindings' not in self.args.GetSpecifiedArgs():
            return []
        policy_bindings = self.args.binauthz_policy_bindings
        return (fleet_messages.PolicyBinding(name=binding['name']) for binding in policy_bindings)

    def OperationRef(self) -> resources.Resource:
        """Parses resource argument operation."""
        return self.args.CONCEPTS.operation.Parse()

    def Project(self) -> str:
        return self.args.project

    def Location(self) -> str:
        return self.args.location

    def Async(self) -> bool:
        """Parses --async flag.

    The internal representation of --async is set to args.async_, defined in
    calliope/base.py file.

    Returns:
      bool, True if specified, False if unspecified.
    """
        return self.args.async_