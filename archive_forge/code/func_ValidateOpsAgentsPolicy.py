from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def ValidateOpsAgentsPolicy(policy):
    """Validates semantics of an Ops agents policy.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is an OpsAgentPolicy object.

  Args:
    policy: ops_agents.OpsAgentPolicy. The policy that manages Ops agents.

  Raises:
    PolicyValidationMultiError that contains a list of validation
    errors from the following list.
    * AgentTypesUniquenessError:
      Multiple agents with the same type are specified.
    * AgentTypesConflictError:
      More than one agent type is specified when there is already a type
      ops-agent.
    * AgentVersionInvalidFormatError:
      Agent version format is invalid.
    * AgentVersionAndEnableAutoupgradeConflictError:
      Agent version is pinned but autoupgrade is enabled.
    * OsTypesMoreThanOneError:
      More than one OS types are specified.
    * OsTypeNotSupportedError:
      The combination of the OS short name and version is not supported.
    * OSTypeNotSupportedByAgentTypeError:
      The combination of the OS short name and agent type is not supported.
  """
    errors = _ValidateAgentRules(policy.agent_rules) + _ValidateOsTypes(policy.assignment.os_types) + _ValidateAgentRulesAndOsTypes(policy.agent_rules, policy.assignment.os_types)
    if errors:
        raise exceptions.PolicyValidationMultiError(errors)
    log.debug('Ops Agents policy validation passed.')