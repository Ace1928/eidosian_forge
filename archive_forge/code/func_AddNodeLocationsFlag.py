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
def AddNodeLocationsFlag(parser):
    parser.add_argument('--node-locations', type=arg_parsers.ArgList(min_length=1), metavar='ZONE', help="The set of zones in which the specified node footprint should be replicated.\nAll zones must be in the same region as the cluster's master(s), specified by\nthe `-location`, `--zone`, or `--region` flag. Additionally, for zonal clusters,\n`--node-locations` must contain the cluster's primary zone. If not specified,\nall nodes will be in the cluster's primary zone (for zonal clusters) or spread\nacross three randomly chosen zones within the cluster's region (for regional\nclusters).\n\nNote that `NUM_NODES` nodes will be created in each zone, such that if you\nspecify `--num-nodes=4` and choose two locations, 8 nodes will be created.\n\nMultiple locations can be specified, separated by commas. For example:\n\n  $ {command} example-cluster --location us-central1-a --node-locations us-central1-a,us-central1-b\n")