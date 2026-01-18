from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import firewalls_utils
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.firewall_rules import flags
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.core.console import progress_tracker
def _CreateFirewall(self, holder, args):
    client = holder.client
    if args.rules and args.allow:
        raise firewalls_utils.ArgumentValidationError('Can NOT specify --rules and --allow in the same request.')
    if bool(args.action) ^ bool(args.rules):
        raise firewalls_utils.ArgumentValidationError('Must specify --rules with --action.')
    allowed = firewalls_utils.ParseRules(args.allow, client.messages, firewalls_utils.ActionType.ALLOW)
    network_ref = self.NETWORK_ARG.ResolveAsResource(args, holder.resources)
    firewall_ref = self.FIREWALL_RULE_ARG.ResolveAsResource(args, holder.resources)
    firewall = client.messages.Firewall(allowed=allowed, name=firewall_ref.Name(), description=args.description, network=network_ref.SelfLink(), sourceRanges=args.source_ranges, sourceTags=args.source_tags, targetTags=args.target_tags)
    if args.disabled is not None:
        firewall.disabled = args.disabled
    firewall.direction = None
    if args.direction and args.direction in ['EGRESS', 'OUT']:
        firewall.direction = client.messages.Firewall.DirectionValueValuesEnum.EGRESS
    else:
        firewall.direction = client.messages.Firewall.DirectionValueValuesEnum.INGRESS
    firewall.priority = args.priority
    firewall.destinationRanges = args.destination_ranges
    allowed = []
    denied = []
    if not args.action:
        allowed = firewalls_utils.ParseRules(args.allow, client.messages, firewalls_utils.ActionType.ALLOW)
    elif args.action == 'ALLOW':
        allowed = firewalls_utils.ParseRules(args.rules, client.messages, firewalls_utils.ActionType.ALLOW)
    elif args.action == 'DENY':
        denied = firewalls_utils.ParseRules(args.rules, client.messages, firewalls_utils.ActionType.DENY)
    firewall.allowed = allowed
    firewall.denied = denied
    firewall.sourceServiceAccounts = args.source_service_accounts
    firewall.targetServiceAccounts = args.target_service_accounts
    if args.IsSpecified('logging_metadata') and (not args.enable_logging):
        raise exceptions.InvalidArgumentException('--logging-metadata', 'cannot toggle logging metadata if logging is not enabled.')
    if args.IsSpecified('enable_logging'):
        log_config = client.messages.FirewallLogConfig(enable=args.enable_logging)
        if args.IsSpecified('logging_metadata'):
            log_config.metadata = flags.GetLoggingMetadataArg(client.messages).GetEnumForChoice(args.logging_metadata)
        firewall.logConfig = log_config
    return (firewall, firewall_ref.project)