from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateAssignment(messages, assignment_group_labels, assignment_os_types, assignment_zones, assignment_instances):
    """Creates a Assignment message from its components."""
    return messages.Assignment(groupLabels=_CreateGroupLabel(messages, assignment_group_labels), zones=assignment_zones or [], instances=assignment_instances or [], osTypes=_CreateOstypes(messages, assignment_os_types))