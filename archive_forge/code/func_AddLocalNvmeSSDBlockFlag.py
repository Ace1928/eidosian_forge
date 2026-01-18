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
def AddLocalNvmeSSDBlockFlag(parser, for_node_pool=False, hidden=False, help_text=''):
    """Adds a --local-nvme-ssd-block flag to the given parser."""
    help_text += "Adds the requested local SSDs on all nodes in default node pool(s) in the new cluster.\n\nExamples:\n\n  $ {{command}} {0} --local-nvme-ssd-block count=2\n\n'count' must be between 1-8\n\n\nNew nodes, including ones created by resize or recreate, will have these local SSDs.\n\nFor first- and second-generation machine types, a nonzero count field is\nrequired for local ssd to be configured. For third-generation machine types, the\ncount field is optional because the count is inferred from the machine type.\n\nSee https://cloud.google.com/compute/docs/disks/local-ssd for more information.\n".format('node-pool-1 --cluster=example cluster' if for_node_pool else 'example_cluster')
    parser.add_argument('--local-nvme-ssd-block', help=help_text, hidden=hidden, nargs='?', type=arg_parsers.ArgDict(spec={'count': int}))