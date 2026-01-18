from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profile_groups import spg_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_security import spg_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class CreateProfileGroup(base.CreateCommand):
    """Create a new Security Profile Group."""

    @classmethod
    def Args(cls, parser):
        spg_flags.AddSecurityProfileGroupResource(parser, cls.ReleaseTrack())
        spg_flags.AddProfileGroupDescription(parser)
        spg_flags.AddThreatPreventionProfileResource(parser, cls.ReleaseTrack(), required=True)
        labels_util.AddCreateLabelsFlags(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        base.ASYNC_FLAG.SetDefault(parser, False)

    def Run(self, args):
        client = spg_api.Client(self.ReleaseTrack())
        security_profile_group = args.CONCEPTS.security_profile_group.Parse()
        security_profile = args.CONCEPTS.threat_prevention_profile.Parse()
        description = args.description
        is_async = args.async_
        labels = labels_util.ParseCreateArgs(args, client.messages.SecurityProfileGroup.LabelsValue)
        if args.location != 'global':
            raise core_exceptions.Error('Only `global` location is supported, but got: %s' % args.location)
        response = client.CreateSecurityProfileGroup(security_profile_group_name=security_profile_group.RelativeName(), security_profile_group_id=security_profile_group.Name(), parent=security_profile_group.Parent().RelativeName(), description=description, threat_prevention_profile=security_profile.RelativeName(), labels=labels)
        if is_async:
            operation_id = response.name
            log.status.Print('Check for operation completion status using operation ID:', operation_id)
            return response
        return client.WaitForOperation(operation_ref=client.GetOperationsRef(response), message='Waiting for security-profile-group [{}] to be created'.format(security_profile_group.RelativeName()), has_result=True)