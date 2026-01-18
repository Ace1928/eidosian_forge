from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class AgentVersionInvalidFormatError(exceptions.PolicyValidationError):
    """Raised when agent version format is invalid."""

    def __init__(self, version):
        super(AgentVersionInvalidFormatError, self).__init__('The agent version [{}] is not allowed. Expected values: [latest], [current-major], or anything in the format of [MAJOR_VERSION.MINOR_VERSION.PATCH_VERSION] or [MAJOR_VERSION.*.*].'.format(version))