from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgSpCreation(parser):
    """Adds the argument for security policy creaton."""
    parser.add_argument('--display-name', help='A textual name of the security policy.')
    group = parser.add_group(required=True, mutex=True)
    group.add_argument('--organization', help='Organization in which the organization security policy is to be created.')
    group.add_argument('--folder', help='Folder in which the organization security policy is to be created.')
    parser.add_argument('--description', help='An optional, textual description for the organization security policy.')