from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddReplicationForceArg(parser):
    """Adds the --force arg to the arg parser."""
    parser.add_argument('--force', action='store_true', help='Indicates whether to stop replication forcefully while data transfer is in progress.\n      Warning! if force is true, this will abort any current transfers and can lead to data loss due to partial transfer.\n      If force is false, stop replication will fail while data transfer is in progress and you will need to retry later.\n      ')