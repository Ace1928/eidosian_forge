from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddEphemeralStorageLocalSSDFlag(parser, hidden=False, for_node_pool=False, help_text=''):
    """Adds --ephemeral-storage-local-ssd flag to the parser."""
    help_text += "Parameters for the ephemeral storage filesystem.\nIf unspecified, ephemeral storage is backed by the boot disk.\n\nExamples:\n\n  $ {{command}} {0} --ephemeral-storage-local-ssd count=2\n\n'count' specifies the number of local SSDs to use to back ephemeral\nstorage. Local SDDs use NVMe interfaces. For first- and second-generation\nmachine types, a nonzero count field is required for local ssd to be configured.\nFor third-generation machine types, the count field is optional because the count\nis inferred from the machine type.\n\nSee https://cloud.google.com/compute/docs/disks/local-ssd for more information.\n".format('node-pool-1 --cluster=example cluster' if for_node_pool else 'example_cluster')
    parser.add_argument('--ephemeral-storage-local-ssd', help=help_text, hidden=hidden, nargs='?', type=arg_parsers.ArgDict(spec={'count': int}))