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
def AddAddonsFlagsWithOptions(parser, addon_options):
    """Adds the --addons flag to the parser with the given addon options."""
    visible_addon_options = [addon for addon in addon_options if addon not in [api_adapter.APPLICATIONMANAGER, api_adapter.STATEFULHA, api_adapter.PARALLELSTORECSIDRIVER]]
    visible_addon_options += api_adapter.VISIBLE_CLOUDRUN_ADDONS
    parser.add_argument('--addons', type=arg_parsers.ArgList(choices=addon_options + api_adapter.CLOUDRUN_ADDONS, visible_choices=visible_addon_options), metavar='ADDON', help='Addons\n(https://cloud.google.com/kubernetes-engine/docs/reference/rest/v1/projects.locations.clusters#Cluster.AddonsConfig)\nare additional Kubernetes cluster components. Addons specified by this flag will\nbe enabled. The others will be disabled. Default addons: {0}.\nThe Istio addon is deprecated and removed.\nFor more information and migration, see https://cloud.google.com/istio/docs/istio-on-gke/migrate-to-anthos-service-mesh.\n'.format(', '.join(api_adapter.DEFAULT_ADDONS)))