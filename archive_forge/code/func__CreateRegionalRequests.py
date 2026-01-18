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
def _CreateRegionalRequests(self, client, resources, args, forwarding_rule_ref):
    """Create a regionally scoped request."""
    is_psc_ilb = False
    if hasattr(args, 'target_service_attachment') and args.target_service_attachment:
        if not self._support_target_service_attachment:
            raise exceptions.InvalidArgumentException('--target-service-attachment', 'Private Service Connect for ILB (the target-service-attachment option) is not supported in this API version.')
        else:
            is_psc_ilb = True
    target_ref, region_ref = utils.GetRegionalTarget(client, resources, args, forwarding_rule_ref, include_regional_tcp_proxy=self._support_regional_tcp_proxy, include_target_service_attachment=self._support_target_service_attachment)
    if not args.region and region_ref:
        args.region = region_ref
    protocol = self.ConstructProtocol(client.messages, args)
    address = self._ResolveAddress(resources, args, compute_flags.compute_scope.ScopeEnum.REGION, forwarding_rule_ref)
    load_balancing_scheme = _GetLoadBalancingScheme(args, client.messages, is_psc_ilb)
    if is_psc_ilb and load_balancing_scheme:
        raise exceptions.InvalidArgumentException('--load-balancing-scheme', 'The --load-balancing-scheme flag is not allowed for PSC-ILB forwarding rules.')
    if load_balancing_scheme == client.messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.INTERNAL:
        if args.port_range:
            raise fw_exceptions.ArgumentError('You cannot specify [--port-range] for a forwarding rule whose [--load-balancing-scheme] is internal, please use [--ports] flag instead.')
    if load_balancing_scheme == client.messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.INTERNAL_SELF_MANAGED:
        raise fw_exceptions.ArgumentError('You cannot specify an INTERNAL_SELF_MANAGED [--load-balancing-scheme] for a regional forwarding rule.')
    forwarding_rule = client.messages.ForwardingRule(description=args.description, name=forwarding_rule_ref.Name(), IPAddress=address, IPProtocol=protocol, networkTier=_ConstructNetworkTier(client.messages, args), loadBalancingScheme=load_balancing_scheme)
    if self._support_source_ip_range and args.source_ip_ranges:
        forwarding_rule.sourceIpRanges = args.source_ip_ranges
    self._ProcessCommonArgs(client, resources, args, forwarding_rule_ref, forwarding_rule)
    ports_all_specified, range_list = _ExtractPortsAndAll(args.ports)
    if target_ref.Collection() == 'compute.regionBackendServices':
        forwarding_rule.backendService = target_ref.SelfLink()
        forwarding_rule.target = None
    else:
        forwarding_rule.backendService = None
        forwarding_rule.target = target_ref.SelfLink()
    if (target_ref.Collection() == 'compute.regionBackendServices' or target_ref.Collection() == 'compute.targetInstances') and args.load_balancing_scheme == 'INTERNAL':
        if ports_all_specified:
            forwarding_rule.allPorts = True
        elif range_list:
            forwarding_rule.ports = [six.text_type(p) for p in _GetPortList(range_list)]
    elif (target_ref.Collection() == 'compute.regionTargetHttpProxies' or target_ref.Collection() == 'compute.regionTargetHttpsProxies') and args.load_balancing_scheme == 'INTERNAL':
        forwarding_rule.ports = [six.text_type(p) for p in _GetPortList(range_list)]
    elif args.load_balancing_scheme == 'INTERNAL':
        raise exceptions.InvalidArgumentException('--load-balancing-scheme', 'Only target instances and backend services should be specified as a target for internal load balancing.')
    elif args.load_balancing_scheme == 'INTERNAL_MANAGED':
        forwarding_rule.portRange = _MakeSingleUnifiedPortRange(args.port_range, range_list)
    elif args.load_balancing_scheme == 'EXTERNAL_MANAGED':
        forwarding_rule.portRange = _MakeSingleUnifiedPortRange(args.port_range, range_list)
    elif target_ref.Collection() == 'compute.regionBackendServices' and (args.load_balancing_scheme == 'EXTERNAL' or not args.load_balancing_scheme):
        if ports_all_specified:
            forwarding_rule.allPorts = True
        elif range_list:
            if len(range_list) > 1:
                forwarding_rule.ports = [six.text_type(p) for p in _GetPortList(range_list)]
            else:
                forwarding_rule.portRange = six.text_type(range_list[0])
        elif args.port_range:
            forwarding_rule.portRange = _MakeSingleUnifiedPortRange(args.port_range, range_list)
    elif (target_ref.Collection() == 'compute.targetPool' or target_ref.Collection() == 'compute.targetInstances') and (args.load_balancing_scheme == 'EXTERNAL' or not args.load_balancing_scheme):
        if ports_all_specified:
            forwarding_rule.allPorts = True
        else:
            forwarding_rule.portRange = _MakeSingleUnifiedPortRange(args.port_range, range_list)
    else:
        forwarding_rule.portRange = _MakeSingleUnifiedPortRange(args.port_range, range_list)
    if hasattr(args, 'service_label'):
        forwarding_rule.serviceLabel = args.service_label
    if self._support_global_access and args.IsSpecified('allow_global_access'):
        forwarding_rule.allowGlobalAccess = args.allow_global_access
    if args.IsSpecified('allow_psc_global_access'):
        forwarding_rule.allowPscGlobalAccess = args.allow_psc_global_access
    if self._support_ip_collection and args.ip_collection:
        forwarding_rule.ipCollection = flags.IP_COLLECTION_ARG.ResolveAsResource(args, resources).SelfLink()
    if self._support_disable_automate_dns_zone and args.IsSpecified('disable_automate_dns_zone'):
        forwarding_rule.noAutomateDnsZone = args.disable_automate_dns_zone
    if hasattr(args, 'is_mirroring_collector'):
        forwarding_rule.isMirroringCollector = args.is_mirroring_collector
    if hasattr(args, 'service_directory_registration') and args.service_directory_registration:
        if is_psc_ilb:
            match = re.match('^projects/([^/]+)/locations/([^/]+)/namespaces/([^/]+)$', args.service_directory_registration)
            if not match:
                raise exceptions.InvalidArgumentException('--service-directory-registration', 'If set, must be of the form projects/PROJECT/locations/REGION/namespaces/NAMESPACE')
            project = match.group(1)
            region = match.group(2)
            if project != forwarding_rule_ref.project or region != forwarding_rule_ref.region:
                raise exceptions.InvalidArgumentException('--service-directory-registration', 'Service Directory registration must be in the same project and region as the forwarding rule.')
            sd_registration = client.messages.ForwardingRuleServiceDirectoryRegistration(namespace=match.group(3))
            forwarding_rule.serviceDirectoryRegistrations.append(sd_registration)
        else:
            if not self._support_sd_registration_for_regional:
                raise exceptions.InvalidArgumentException('--service-directory-registration', 'flag is available in one or more alternate release tracks. Try:\n\n  gcloud alpha compute forwarding-rules create --service-directory-registration\n  gcloud beta compute forwarding-rules create --service-directory-registration')
            match = re.match('^projects/([^/]+)/locations/([^/]+)/namespaces/([^/]+)/services/([^/]+)$', args.service_directory_registration)
            if not match:
                raise exceptions.InvalidArgumentException('--service-directory-registration', 'Must be of the form projects/PROJECT/locations/REGION/namespaces/NAMESPACE/services/SERVICE')
            project = match.group(1)
            region = match.group(2)
            if project != forwarding_rule_ref.project or region != forwarding_rule_ref.region:
                raise exceptions.InvalidArgumentException('--service-directory-registration', 'Service Directory registration must be in the same project and region as the forwarding rule.')
            sd_registration = client.messages.ForwardingRuleServiceDirectoryRegistration(namespace=match.group(3), service=match.group(4))
            forwarding_rule.serviceDirectoryRegistrations.append(sd_registration)
    request = client.messages.ComputeForwardingRulesInsertRequest(forwardingRule=forwarding_rule, project=forwarding_rule_ref.project, region=forwarding_rule_ref.region)
    return [(client.apitools_client.forwardingRules, 'Insert', request)]