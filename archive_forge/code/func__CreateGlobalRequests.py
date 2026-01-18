from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions as fw_exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import log
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _CreateGlobalRequests(self, client, resources, args, forwarding_rule_ref):
    """Create a globally scoped request."""
    is_psc_google_apis = False
    if hasattr(args, 'target_google_apis_bundle') and args.target_google_apis_bundle:
        if not self._support_psc_google_apis:
            raise exceptions.InvalidArgumentException('--target-google-apis-bundle', 'Private Service Connect for Google APIs (the target-google-apis-bundle option for forwarding rules) is not supported in this API version.')
        else:
            is_psc_google_apis = True
    sd_registration = None
    if hasattr(args, 'service_directory_registration') and args.service_directory_registration:
        if not is_psc_google_apis:
            raise exceptions.InvalidArgumentException('--service-directory-registration', 'Can only be specified for regional forwarding rules or Private Service Connect forwarding rules targeting a Google APIs bundle.')
        match = re.match('^projects/([^/]+)/locations/([^/]+)(?:/namespaces/([^/]+))?$', args.service_directory_registration)
        if not match:
            raise exceptions.InvalidArgumentException('--service-directory-registration', 'Must be of the form projects/PROJECT/locations/REGION or projects/PROJECT/locations/REGION/namespaces/NAMESPACE')
        project = match.group(1)
        region = match.group(2)
        namespace = match.group(3)
        if project != forwarding_rule_ref.project:
            raise exceptions.InvalidArgumentException('--service-directory-registration', 'Must be in the same project as the forwarding rule.')
        sd_registration = client.messages.ForwardingRuleServiceDirectoryRegistration(serviceDirectoryRegion=region, namespace=namespace)
    ports_all_specified, range_list = _ExtractPortsAndAll(args.ports)
    port_range = _MakeSingleUnifiedPortRange(args.port_range, range_list)
    load_balancing_scheme = _GetLoadBalancingScheme(args, client.messages, is_psc_google_apis)
    if load_balancing_scheme == client.messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.INTERNAL:
        raise fw_exceptions.ArgumentError('You cannot specify internal [--load-balancing-scheme] for a global forwarding rule.')
    if load_balancing_scheme == client.messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.INTERNAL_SELF_MANAGED:
        if not args.target_http_proxy and (not args.target_https_proxy) and (not args.target_grpc_proxy) and (not args.target_tcp_proxy):
            target_error_message_with_tcp = 'You must specify either [--target-http-proxy], [--target-https-proxy], [--target-grpc-proxy] or [--target-tcp-proxy] for an INTERNAL_SELF_MANAGED [--load-balancing-scheme].'
            raise fw_exceptions.ArgumentError(target_error_message_with_tcp)
        if args.subnet:
            raise fw_exceptions.ArgumentError('You cannot specify [--subnet] for an INTERNAL_SELF_MANAGED [--load-balancing-scheme].')
        if not args.address:
            raise fw_exceptions.ArgumentError('You must specify [--address] for an INTERNAL_SELF_MANAGED [--load-balancing-scheme]')
    if is_psc_google_apis:
        rule_name = forwarding_rule_ref.Name()
        if len(rule_name) > 20 or rule_name[0].isdigit() or (not rule_name.isalnum()):
            raise fw_exceptions.ArgumentError('A forwarding rule to Google APIs must have a name that is between  1-20 characters long, alphanumeric, starting with a letter.')
        if port_range:
            raise exceptions.InvalidArgumentException('--ports', '[--ports] is not allowed for PSC-GoogleApis forwarding rules.')
        if load_balancing_scheme:
            raise exceptions.InvalidArgumentException('--load-balancing-scheme', 'The --load-balancing-scheme flag is not allowed for PSC-GoogleApis forwarding rules.')
        if args.target_google_apis_bundle in flags.PSC_GOOGLE_APIS_BUNDLES:
            target_as_str = args.target_google_apis_bundle
        else:
            bundles_list = ', '.join(flags.PSC_GOOGLE_APIS_BUNDLES)
            raise exceptions.InvalidArgumentException('--target-google-apis-bundle', 'The valid values for target-google-apis-bundle are: ' + bundles_list)
    else:
        target_ref = utils.GetGlobalTarget(resources, args)
        target_as_str = target_ref.SelfLink()
        if ports_all_specified:
            raise exceptions.InvalidArgumentException('--ports', '[--ports] cannot be set to ALL for global forwarding rules.')
        if not port_range:
            raise exceptions.InvalidArgumentException('--ports', '[--ports] is required for global forwarding rules.')
    protocol = self.ConstructProtocol(client.messages, args)
    address = self._ResolveAddress(resources, args, compute_flags.compute_scope.ScopeEnum.GLOBAL, forwarding_rule_ref)
    forwarding_rule = client.messages.ForwardingRule(description=args.description, name=forwarding_rule_ref.Name(), IPAddress=address, IPProtocol=protocol, portRange=port_range, target=target_as_str, networkTier=_ConstructNetworkTier(client.messages, args), loadBalancingScheme=load_balancing_scheme)
    self._ProcessCommonArgs(client, resources, args, forwarding_rule_ref, forwarding_rule)
    if sd_registration:
        forwarding_rule.serviceDirectoryRegistrations.append(sd_registration)
    if self._support_global_access and args.IsSpecified('allow_global_access'):
        forwarding_rule.allowGlobalAccess = args.allow_global_access
    request = client.messages.ComputeGlobalForwardingRulesInsertRequest(forwardingRule=forwarding_rule, project=forwarding_rule_ref.project)
    return [(client.apitools_client.globalForwardingRules, 'Insert', request)]