from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import firewalls_utils
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute.firewall_rules import flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class BetaUpdateFirewall(UpdateFirewall):
    """Update a firewall rule."""

    @classmethod
    def Args(cls, parser):
        messages = apis.GetMessagesModule('compute', compute_api.COMPUTE_BETA_API_VERSION)
        cls.FIREWALL_RULE_ARG = flags.FirewallRuleArgument()
        cls.FIREWALL_RULE_ARG.AddArgument(parser, operation_type='update')
        firewalls_utils.AddCommonArgs(parser, for_update=True, with_egress_support=cls.with_egress_firewall, with_service_account=cls.with_service_account)
        firewalls_utils.AddArgsForServiceAccount(parser, for_update=True)
        flags.AddEnableLogging(parser)
        flags.AddLoggingMetadata(parser, messages)