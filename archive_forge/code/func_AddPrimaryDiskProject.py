from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddPrimaryDiskProject(parser, category=None):
    parser.add_argument('--primary-disk-project', category=category, help=_ASYNC_PRIMARY_DISK_PROJECT_EXPLANATION)