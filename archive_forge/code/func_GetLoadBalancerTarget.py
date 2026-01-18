from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from dns import rdatatype
from googlecloudsdk.api_lib.dns import import_util
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def GetLoadBalancerTarget(forwarding_rule, api_version, project):
    """Creates and returns a LoadBalancerTarget for the given forwarding rule name.

  Args:
    forwarding_rule: The name of the forwarding rule followed by '@' followed by
      the scope of the forwarding rule.
    api_version: [str], the api version to use for creating the RecordSet.
    project: The GCP project where the forwarding_rule exists.

  Raises:
    ForwardingRuleNotFound: Either the forwarding rule doesn't exist, or
      multiple forwarding rules present with the same name - across different
      regions.
    UnsupportedLoadBalancingScheme: The requested load balancer uses a load
      balancing scheme that is not supported by Cloud DNS Policy Manager.

  Returns:
    LoadBalancerTarget, the load balancer target for the given forwarding rule.
  """
    compute_client = apis.GetClientInstance('compute', 'v1')
    compute_messages = apis.GetMessagesModule('compute', 'v1')
    dns_messages = apis.GetMessagesModule('dns', api_version)
    load_balancer_target = apis.GetMessagesModule('dns', api_version).RRSetRoutingPolicyLoadBalancerTarget()
    load_balancer_target.project = project
    load_balancer_type = ''
    if len(forwarding_rule.split('@')) == 2:
        name, scope = forwarding_rule.split('@')
        if scope == 'global':
            config = compute_client.globalForwardingRules.Get(compute_messages.ComputeGlobalForwardingRulesGetRequest(project=project, forwardingRule=name))
        else:
            load_balancer_target.region = scope
            config = compute_client.forwardingRules.Get(compute_messages.ComputeForwardingRulesGetRequest(project=project, forwardingRule=name, region=scope))
        if config is None:
            raise ForwardingRuleNotFound("Either the forwarding rule doesn't exist, or multiple forwarding rules are present with the same name - across different regions.")
    else:
        try:
            config = GetLoadBalancerConfigFromUrl(compute_client, compute_messages, forwarding_rule)
            project_match = re.match('.*/projects/([^/]+)/.*', config.selfLink)
            load_balancer_target.project = project_match.group(1)
            if config.region:
                region_match = re.match('.*/regions/(.*)$', config.region)
                load_balancer_target.region = region_match.group(1)
        except (resources.WrongResourceCollectionException, resources.RequiredFieldOmittedException):
            regions = [item.name for item in compute_client.regions.List(compute_messages.ComputeRegionsListRequest(project=project)).items]
            configs = []
            for region in regions:
                configs.extend(compute_client.forwardingRules.List(compute_messages.ComputeForwardingRulesListRequest(filter='name = %s' % forwarding_rule, project=project, region=region)).items)
            configs.extend(compute_client.globalForwardingRules.List(compute_messages.ComputeGlobalForwardingRulesListRequest(filter='name = %s' % forwarding_rule, project=project)).items)
            if not configs:
                raise ForwardingRuleNotFound('The forwarding rule %s was not found.' % forwarding_rule)
            if len(configs) > 1:
                raise ForwardingRuleNotFound('There are multiple forwarding rules present with the same name across different regions. Specify the intended region along with the rule in the format: forwardingrulename@region.')
            config = configs[0]
            if config.region:
                region_match = re.match('.*/regions/(.*)$', config.region)
                load_balancer_target.region = region_match.group(1)
    if config.loadBalancingScheme == compute_messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum('INTERNAL') and config.backendService:
        load_balancer_type = 'regionalL4ilb'
    elif config.loadBalancingScheme == compute_messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum('INTERNAL_MANAGED') and ('/targetHttpProxies/' in config.target or '/targetHttpsProxies/' in config.target):
        if '/regions/' in config.target:
            load_balancer_type = 'regionalL7ilb'
        else:
            load_balancer_type = 'globalL7ilb'
    else:
        raise UnsupportedLoadBalancingScheme('Only Regional internal passthrough Network load balancers and Regional/Global internal Application load balancers are supported at this time.')
    load_balancer_target.ipAddress = config.IPAddress
    compute_tcp_enum = compute_messages.ForwardingRule.IPProtocolValueValuesEnum('TCP')
    ip_protocol = 'tcp' if config.IPProtocol == compute_tcp_enum else 'udp'
    load_balancer_target.networkUrl = config.network
    if config.allPorts:
        load_balancer_target.port = '80'
    elif not config.ports:
        load_balancer_target.port = config.portRange.split('-')[0]
    else:
        load_balancer_target.port = config.ports[0]
    if api_version in ['dev', 'v2']:
        load_balancer_type = util.CamelCaseToSnakeCase(load_balancer_type)
        ip_protocol = util.CamelCaseToSnakeCase(ip_protocol)
    load_balancer_target.ipProtocol = dns_messages.RRSetRoutingPolicyLoadBalancerTarget.IpProtocolValueValuesEnum(ip_protocol)
    load_balancer_target.loadBalancerType = dns_messages.RRSetRoutingPolicyLoadBalancerTarget.LoadBalancerTypeValueValuesEnum(load_balancer_type)
    return load_balancer_target