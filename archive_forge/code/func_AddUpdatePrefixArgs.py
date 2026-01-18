from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddUpdatePrefixArgs(parser):
    parser.add_argument('--announce-prefix', action='store_true', default=False, help='Specify if the prefix will be announced. Default is false.')
    parser.add_argument('--withdraw-prefix', action='store_true', default=False, help='Specify if the prefix will be withdrawn. Default is false.')