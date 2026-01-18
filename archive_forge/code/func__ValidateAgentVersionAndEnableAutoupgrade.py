from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateAgentVersionAndEnableAutoupgrade(version, enable_autoupgrade):
    """Validates that agent version is not pinned when autoupgrade is enabled.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the fields are valid string and boolean.

  Args:
    version: str. The version of agent. Possible formats:
      * "latest"
      * "[MAJOR_VERSION].*.*"
      * "[MAJOR_VERSION].[MINOR_VERSION].[PATCH_VERSION]"
    enable_autoupgrade: bool. Whether autoupgrade is enabled.

  Returns:
    An empty list if the validation passes. A singleton list with the following
    error if the validation fails.
    * AgentVersionAndEnableAutoupgradeConflictError:
      Agent version is pinned but autoupgrade is enabled.
  """
    if _PINNED_VERSION_RE.search(version) and enable_autoupgrade:
        return [AgentVersionAndEnableAutoupgradeConflictError(version)]
    return []