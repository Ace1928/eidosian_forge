from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator, List
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
class FleetFlagParser:
    """Parse flags during fleet command runtime."""

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

    def Fleet(self, existing_fleet=None) -> fleet_messages.Fleet:
        """Fleet resource."""
        fleet = self.messages.Fleet()
        fleet.name = util.FleetResourceName(self.Project())
        fleet.displayName = self._DisplayName()
        if self.release_track == base.ReleaseTrack.ALPHA:
            if existing_fleet is not None:
                fleet.defaultClusterConfig = self._DefaultClusterConfig(existing_fleet.defaultClusterConfig)
            else:
                fleet.defaultClusterConfig = self._DefaultClusterConfig()
        return fleet

    def _DisplayName(self) -> str:
        return self.args.display_name

    def Project(self) -> str:
        return arg_utils.GetFromNamespace(self.args, '--project', use_defaults=True)

    def Async(self) -> bool:
        """Parses --async flag.

    The internal representation of --async is set to args.async_, defined in
    calliope/base.py file.

    Returns:
      bool, True if specified, False if unspecified.
    """
        return self.args.async_

    def _SecurityPostureConfig(self) -> fleet_messages.SecurityPostureConfig:
        ret = self.messages.SecurityPostureConfig()
        ret.mode = self._SecurityPostureMode()
        ret.vulnerabilityMode = self._VulnerabilityModeValueValuesEnum()
        return self.TrimEmpty(ret)

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
        enum_type = fleet_messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum
        mapping = {'disabled': enum_type.VULNERABILITY_DISABLED, 'standard': enum_type.VULNERABILITY_BASIC, 'enterprise': enum_type.VULNERABILITY_ENTERPRISE}
        return mapping[self.args.workload_vulnerability_scanning]

    def _BinaryAuthorizationConfig(self, existing_binauthz=None) -> fleet_messages.BinaryAuthorizationConfig:
        """Construct binauthz config from args."""
        new_binauthz = self.messages.BinaryAuthorizationConfig()
        new_binauthz.evaluationMode = self._EvaluationMode()
        new_binauthz.policyBindings = list(self._PolicyBindings())
        if existing_binauthz is None:
            ret = new_binauthz
        else:
            ret = existing_binauthz
            if new_binauthz.evaluationMode is not None:
                ret.evaluationMode = new_binauthz.evaluationMode
            if new_binauthz.policyBindings is not None:
                ret.policyBindings = new_binauthz.policyBindings
        if ret.policyBindings and (not ret.evaluationMode):
            raise exceptions.InvalidArgumentException('--binauthz-policy-bindings', _PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='binauthz-evaluation-mode', opt='binauthz-policy-bindings'))
        if ret.evaluationMode == fleet_messages.BinaryAuthorizationConfig.EvaluationModeValueValuesEnum.DISABLED:
            ret.policyBindings = []
        return self.TrimEmpty(ret)

    def _EvaluationMode(self) -> fleet_messages.BinaryAuthorizationConfig.EvaluationModeValueValuesEnum:
        """Parses --binauthz-evaluation-mode."""
        if '--binauthz-evaluation-mode' not in self.args.GetSpecifiedArgs():
            return None
        enum_type = self.messages.BinaryAuthorizationConfig.EvaluationModeValueValuesEnum
        mapping = {'disabled': enum_type.DISABLED, 'policy-bindings': enum_type.POLICY_BINDINGS}
        return mapping[self.args.binauthz_evaluation_mode]

    def _PolicyBindings(self) -> Iterator[fleet_messages.PolicyBinding]:
        """Parses --binauthz-policy-bindings."""
        policy_bindings = self.args.binauthz_policy_bindings
        if policy_bindings is not None:
            return (fleet_messages.PolicyBinding(name=binding['name']) for binding in policy_bindings)
        return []

    def _DefaultClusterConfig(self, existing_default_cluster_config=None) -> fleet_messages.DefaultClusterConfig:
        ret = self.messages.DefaultClusterConfig()
        ret.securityPostureConfig = self._SecurityPostureConfig()
        if existing_default_cluster_config is not None:
            ret.binaryAuthorizationConfig = self._BinaryAuthorizationConfig(existing_default_cluster_config.binaryAuthorizationConfig)
        else:
            ret.binaryAuthorizationConfig = self._BinaryAuthorizationConfig()
        return self.TrimEmpty(ret)

    def OperationRef(self) -> resources.Resource:
        """Parses resource argument operation."""
        return self.args.CONCEPTS.operation.Parse()

    def Location(self) -> str:
        return self.args.location

    def PageSize(self) -> int:
        """Returns page size in a list request."""
        return self.args.page_size

    def Limit(self) -> int:
        """Returns limit in a list request."""
        return self.args.limit