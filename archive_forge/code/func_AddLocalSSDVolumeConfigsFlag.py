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
def AddLocalSSDVolumeConfigsFlag(parser, for_node_pool=False, help_text=''):
    """Adds a --local-ssd-volumes flag to the given parser."""
    help_text += "Adds the requested local SSDs on all nodes in default node pool(s) in the new cluster.\n\nExamples:\n\n  $ {{command}} {0} --local-ssd-volumes count=2,type=nvme,format=fs\n\n'count' must be between 1-8\n\n'type' must be either scsi or nvme\n\n'format' must be either fs or block\n\nNew nodes, including ones created by resize or recreate, will have these local SSDs.\n\nLocal SSDs have a fixed 375 GB capacity per device. The number of disks that\ncan be attached to an instance is limited by the maximum number of disks\navailable on a machine, which differs by compute zone. See\nhttps://cloud.google.com/compute/docs/disks/local-ssd for more information.\n".format('node-pool-1 --cluster=example-cluster' if for_node_pool else 'example_cluster')
    count_validator = arg_parsers.RegexpValidator('^[1-8]$', 'Count must be a number between 1 and 8')
    type_validator = arg_parsers.RegexpValidator('^(scsi|nvme)$', 'Type must be either "scsi" or "nvme"')
    format_validator = arg_parsers.RegexpValidator('^(fs|block)$', 'Format must be either "fs" or "block"')
    parser.add_argument('--local-ssd-volumes', metavar='[count=COUNT],[type=TYPE],[format=FORMAT]', type=arg_parsers.ArgDict(spec={'count': count_validator, 'type': type_validator, 'format': format_validator}, required_keys=['count', 'type', 'format'], max_length=3), action='append', help=help_text)