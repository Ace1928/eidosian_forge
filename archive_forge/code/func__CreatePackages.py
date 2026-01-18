from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreatePackages(messages, agent_rules, os_type):
    """Create OS Agent guest policy packages from Ops Agent policy agent field."""
    packages = []
    for agent_rule in agent_rules or []:
        if agent_rule.type is agent_policy.OpsAgentPolicy.AgentRule.Type.LOGGING:
            packages.append(_CreatePackage(messages, 'google-fluentd', agent_rule.package_state, agent_rule.enable_autoupgrade))
            packages.append(_CreatePackage(messages, 'google-fluentd-catch-all-config', agent_rule.package_state, agent_rule.enable_autoupgrade))
            if os_type.short_name not in _APT_OS:
                packages.append(_CreatePackage(messages, 'google-fluentd-start-service', agent_rule.package_state, agent_rule.enable_autoupgrade))
        if agent_rule.type is agent_policy.OpsAgentPolicy.AgentRule.Type.METRICS:
            packages.append(_CreatePackage(messages, 'stackdriver-agent', agent_rule.package_state, agent_rule.enable_autoupgrade))
            if os_type.short_name not in _APT_OS:
                packages.append(_CreatePackage(messages, 'stackdriver-agent-start-service', agent_rule.package_state, agent_rule.enable_autoupgrade))
        if agent_rule.type is agent_policy.OpsAgentPolicy.AgentRule.Type.OPS_AGENT:
            packages.append(_CreatePackage(messages, 'google-cloud-ops-agent', agent_rule.package_state, agent_rule.enable_autoupgrade))
    return packages