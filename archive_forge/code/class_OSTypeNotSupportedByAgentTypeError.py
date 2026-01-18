from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class OSTypeNotSupportedByAgentTypeError(exceptions.PolicyValidationError):
    """Raised when the OS short name and agent type combination is not supported."""

    def __init__(self, short_name, agent_type):
        super(OSTypeNotSupportedByAgentTypeError, self).__init__('The combination of short name [{}] and agent type [{}] is not supported. The supported combinations are: {}.'.format(short_name, agent_type, json.dumps(_SUPPORTED_OS_SHORT_NAMES_AND_AGENT_TYPES)))