from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class OsTypesMoreThanOneError(exceptions.PolicyValidationError):
    """Raised when more than one OS types are specified."""

    def __init__(self):
        super(OsTypesMoreThanOneError, self).__init__('Only one OS type is allowed in the instance filters.')