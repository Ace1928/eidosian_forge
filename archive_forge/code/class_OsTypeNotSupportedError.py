from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class OsTypeNotSupportedError(exceptions.PolicyValidationError):
    """Raised when the OS short name and version combination is not supported."""

    def __init__(self, short_name, version):
        super(OsTypeNotSupportedError, self).__init__('The combination of short name [{}] and version [{}] is not supported. The supported versions are: {}.'.format(short_name, version, json.dumps(_SUPPORTED_OS_SHORT_NAMES_AND_VERSIONS)))