from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profiles.threat_prevention import sp_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_security import sp_flags
from googlecloudsdk.core import exceptions as core_exceptions
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ListOverrides(base.DescribeCommand):
    """List overrides of Threat Prevention Profile."""

    @classmethod
    def Args(cls, parser):
        sp_flags.AddSecurityProfileResource(parser, cls.ReleaseTrack())

    def Run(self, args):
        client = sp_api.Client(self.ReleaseTrack())
        security_profile = args.CONCEPTS.security_profile.Parse()
        if args.location != 'global':
            raise core_exceptions.Error('Only `global` location is supported, but got: %s' % args.location)
        return client.ListOverrides(security_profile.RelativeName())