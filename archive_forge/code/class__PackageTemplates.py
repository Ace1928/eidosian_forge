from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
class _PackageTemplates(collections.namedtuple('_PackageTemplates', ('repo', 'clear_prev_repo'))):
    pass