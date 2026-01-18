from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions as ops_agents_exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents.converters import guest_policy_to_ops_agents_policy_converter as to_ops_agents
from googlecloudsdk.api_lib.compute.instances.ops_agents.converters import os_policy_assignment_to_cloud_ops_agents_policy_converter as to_ops_agents_policy
from googlecloudsdk.api_lib.compute.instances.ops_agents.validators import guest_policy_validator
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.instances.ops_agents.policies import parser_utils
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class DescribeOsConfig(base.DescribeCommand):
    """Describe a Google Cloud's operations suite agents (Ops Agents) policy.

  *{command}* describes a policy that facilitates agent management across
  Compute Engine instances based on user specified instance filters. This policy
  installs, specifies versioning, enables autoupgrade, and removes Ops Agents.

  The command returns the content of one policy. For instance:

    agent_rules:
    - enable_autoupgrade: true
      package_state: installed
      type: ops-agent
      version: latest
    assignment:
      group_labels:
      - app: myapp
        env: prod
      os_types:
      - short_name: ubuntu
        version: '18.04'
      zones:
      - us-central1-a
    create_time: '2021-02-02T02:10:25.344Z'
    description: A test policy to install agents
    etag: <ETAG>
    id: projects/<PROJECT_NUMBER>/guestPolicies/ops-agents-test-policy
    update_time: '2021-02-02T02:10:25.344Z'

  If no policies are found, it returns a ``NOT_FOUND'' error.
  """
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': "          To describe an Ops Agents policy named ``ops-agents-test-policy'' in\n          the current project, run:\n\n            $ {command} ops-agents-test-policy\n          "}

    @staticmethod
    def Args(parser):
        """See base class."""
        parser_utils.AddSharedArgs(parser)

    def Run(self, args):
        """See base class."""
        release_track = self.ReleaseTrack()
        project = properties.VALUES.core.project.GetOrFail()
        guest_policy_uri_path = osconfig_command_utils.GetGuestPolicyUriPath('projects', project, args.POLICY_ID)
        client = osconfig_api_utils.GetClientInstance(release_track, api_version_override='v1beta')
        service = client.projects_guestPolicies
        messages = osconfig_api_utils.GetClientMessages(release_track, api_version_override='v1beta')
        get_request = messages.OsconfigProjectsGuestPoliciesGetRequest(name=guest_policy_uri_path)
        try:
            get_response = service.Get(get_request)
        except apitools_exceptions.HttpNotFoundError:
            raise ops_agents_exceptions.PolicyNotFoundError(policy_id=args.POLICY_ID)
        if not guest_policy_validator.IsOpsAgentPolicy(get_response):
            raise ops_agents_exceptions.PolicyNotFoundError(policy_id=args.POLICY_ID)
        try:
            ops_agents_policy = to_ops_agents.ConvertGuestPolicyToOpsAgentPolicy(get_response)
        except calliope_exceptions.BadArgumentException:
            raise ops_agents_exceptions.PolicyMalformedError(policy_id=args.POLICY_ID)
        return ops_agents_policy