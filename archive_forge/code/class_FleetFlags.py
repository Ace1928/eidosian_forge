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
class FleetFlags:
    """Add flags to the fleet command surface."""

    def __init__(self, parser: parser_arguments.ArgumentInterceptor):
        self._parser = parser

    @property
    def parser(self):
        return self._parser

    @property
    def command_name(self) -> List[str]:
        """Returns the command name.

    This provides information on the command track, command group, and the
    action.

    Returns:
      A list of command, for `gcloud alpha container fleet operations describe`,
      it returns `['gcloud', 'alpha', 'container', 'fleet', 'operations',
      'describe']`.
    """
        return self.parser.command_name

    @property
    def action(self) -> str:
        return self.command_name[-1]

    @property
    def release_track(self) -> base.ReleaseTrack:
        """Returns the release track from the given command name."""
        if self.command_name[1] == 'alpha':
            return base.ReleaseTrack.ALPHA
        elif self.command_name[1] == 'beta':
            return base.ReleaseTrack.BETA
        else:
            return base.ReleaseTrack.GA

    def AddAsync(self):
        base.ASYNC_FLAG.AddToParser(self.parser)

    def AddDisplayName(self):
        self.parser.add_argument('--display-name', type=str, help='Display name of the fleet to be created (optional). 4-30 characters, alphanumeric and [ \'"!-] only.')

    def AddDefaultClusterConfig(self):
        default_cluster_config_group = self.parser.add_group(help='Default cluster configurations to apply across the fleet.')
        self._AddSecurityPostureConfig(default_cluster_config_group)
        self._AddBinaryAuthorizationConfig(default_cluster_config_group)

    def _AddSecurityPostureConfig(self, default_cluster_config_group: parser_arguments.ArgumentInterceptor):
        security_posture_config_group = default_cluster_config_group.add_group(help='Security posture config.')
        self._AddSecurityPostureMode(security_posture_config_group)
        self._AddWorkloadVulnerabilityScanningMode(security_posture_config_group)

    def _AddSecurityPostureMode(self, security_posture_config_group: parser_arguments.ArgumentInterceptor):
        security_posture_config_group.add_argument('--security-posture', choices=['disabled', 'standard'], default=None, help=textwrap.dedent('          To apply standard security posture to clusters in the fleet,\n\n            $ {command} --security-posture=standard\n\n          '))

    def _AddWorkloadVulnerabilityScanningMode(self, security_posture_config_group: parser_arguments.ArgumentInterceptor):
        security_posture_config_group.add_argument('--workload-vulnerability-scanning', choices=['disabled', 'standard', 'enterprise'], default=None, help=textwrap.dedent('            To apply standard vulnerability scanning to clusters in the fleet,\n\n              $ {command} --workload-vulnerability-scanning=standard\n\n            '))

    def _AddBinaryAuthorizationConfig(self, default_cluster_config_group: parser_arguments.ArgumentInterceptor):
        binary_authorization_config_group = default_cluster_config_group.add_group(help='Binary Authorization config.')
        self._AddBinauthzEvaluationMode(binary_authorization_config_group)
        self._AddBinauthzPolicyBindings(binary_authorization_config_group)

    def _AddBinauthzEvaluationMode(self, binary_authorization_config_group: parser_arguments.ArgumentInterceptor):
        binary_authorization_config_group.add_argument('--binauthz-evaluation-mode', choices=['disabled', 'policy-bindings'], type=lambda x: x.replace('_', '-').lower(), default=None, help=textwrap.dedent('          Configure binary authorization mode for clusters to onboard the fleet,\n\n            $ {command} --binauthz-evaluation-mode=policy-bindings\n\n          '))

    def _AddBinauthzPolicyBindings(self, binary_authorization_config_group: parser_arguments.ArgumentInterceptor):
        platform_policy_type = arg_parsers.RegexpValidator(_BINAUTHZ_GKE_POLICY_REGEX, 'GKE policy resource names have the following format: `projects/{project_number}/platforms/gke/policies/{policy_id}`')
        binary_authorization_config_group.add_argument('--binauthz-policy-bindings', default=None, action='append', metavar='name=BINAUTHZ_POLICY', help=textwrap.dedent('          The relative resource name of the Binary Authorization policy to audit\n          and/or enforce. GKE policies have the following format:\n          `projects/{project_number}/platforms/gke/policies/{policy_id}`.'), type=arg_parsers.ArgDict(spec={'name': platform_policy_type}, required_keys=['name'], max_length=1))

    def _OperationResourceSpec(self):
        return concepts.ResourceSpec('gkehub.projects.locations.operations', resource_name='operation', api_version=util.VERSION_MAP[self.release_track], locationsId=self._LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)

    def AddOperationResourceArg(self):
        concept_parsers.ConceptParser.ForResource('operation', self._OperationResourceSpec(), group_help='operation to {}.'.format(self.action), required=True).AddToParser(self.parser)
        self.parser.set_defaults(location='global')

    def _LocationAttributeConfig(self):
        """Gets Google Cloud location resource attribute."""
        return concepts.ResourceParameterAttributeConfig(name='location', help_text='Google Cloud location for the {resource}.')

    def AddLocation(self):
        self.parser.add_argument('--location', type=str, help='The location name.', default='-')