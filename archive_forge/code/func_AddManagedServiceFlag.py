from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.util.args import labels_util
def AddManagedServiceFlag(parser):
    """Adds the managed service flag."""
    parser.add_argument('--managed-service', help='      The name of a pre-existing Google Managed Service.\n      ')