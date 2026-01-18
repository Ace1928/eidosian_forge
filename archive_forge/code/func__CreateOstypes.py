from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateOstypes(messages, assignment_os_types):
    os_types = []
    for assignment_os_type in assignment_os_types or []:
        os_type = messages.AssignmentOsType(osShortName=assignment_os_type.short_name, osVersion=assignment_os_type.version)
        os_types.append(os_type)
    return os_types