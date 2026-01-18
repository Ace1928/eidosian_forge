from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddChainArg(parser):
    parser.add_argument('--chain-name', help='Create the new snapshot in the snapshot chain labeled with the specified name.\n          The chain name must be 1-63 characters long and comply with RFC1035.\n          Use this flag only if you are an advanced service owner who needs\n          to create separate snapshot chains, for example, for chargeback tracking.\n          When you describe your snapshot resource, this field is visible only\n          if it has a non-empty value.')