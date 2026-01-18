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
def AddIstioConfigFlag(parser, suppressed=False):
    """Adds --istio-config flag to the parser.

  Args:
    parser: A given parser.
    suppressed: Whether or not to suppress help text.
  """
    help_text = 'Configurations for Istio addon, requires --addons contains Istio for create,\nor --update-addons Istio=ENABLED for update.\n\n*auth*::: (Optional) Type of auth MTLS_PERMISSIVE or MTLS_STRICT.\n\nExamples:\n\n  $ {command} example-cluster --istio-config=auth=MTLS_PERMISSIVE\n'
    parser.add_argument('--istio-config', metavar='auth=MTLS_PERMISSIVE', type=arg_parsers.ArgDict(spec={'auth': lambda x: x.upper()}), action=actions.DeprecationAction('--istio-config', error='The `--istio-config` flag is no longer supported. For more information and migration, see https://cloud.google.com/istio/docs/istio-on-gke/migrate-to-anthos-service-mesh.', removed=True), help=help_text, hidden=suppressed)