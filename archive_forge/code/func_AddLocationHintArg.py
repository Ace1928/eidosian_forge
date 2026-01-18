from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddLocationHintArg(parser):
    parser.add_argument('--location-hint', hidden=True, help='      Used by internal tools to control sub-zone location of the disk.\n      ')