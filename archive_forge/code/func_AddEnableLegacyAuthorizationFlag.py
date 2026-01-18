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
def AddEnableLegacyAuthorizationFlag(parser, hidden=False):
    """Adds a --enable-legacy-authorization flag to parser."""
    help_text = 'Enables the legacy ABAC authentication for the cluster.\nUser rights are granted through the use of policies which combine attributes\ntogether. For a detailed look at these properties and related formats, see\nhttps://kubernetes.io/docs/admin/authorization/abac/. To use RBAC permissions\ninstead, create or update your cluster with the option\n`--no-enable-legacy-authorization`.\n'
    parser.add_argument('--enable-legacy-authorization', action='store_true', default=None, hidden=hidden, help=help_text)