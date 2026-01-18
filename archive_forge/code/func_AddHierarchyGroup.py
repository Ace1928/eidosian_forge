from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
def AddHierarchyGroup(parser):
    """Adds necessary hierarchy flags for discover command parser."""
    hierarchy_parser = parser.add_group(mutex=True)
    hierarchy_parser.add_argument('--full-hierarchy', help='Whether to retrieve the full hierarchy of data objects (TRUE) or only the current level (FALSE).', action='store_true')
    hierarchy_parser.add_argument('--hierarchy-depth', help='The number of hierarchy levels below the current level to be retrieved.')