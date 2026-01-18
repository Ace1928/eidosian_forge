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
def AddSandboxFlag(parser, hidden=False):
    """Adds a --sandbox flag to the given parser.

  Args:
    parser: A given parser.
    hidden: Whether or not to hide the help text.
  """
    type_validator = arg_parsers.RegexpValidator('^gvisor$', 'Type must be "gvisor"')
    parser.add_argument('--sandbox', type=arg_parsers.ArgDict(spec={'type': type_validator}, required_keys=['type'], max_length=1), metavar='type=TYPE', hidden=hidden, help='Enables the requested sandbox on all nodes in the node pool.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --sandbox="type=gvisor"\n\nThe only supported type is \'gvisor\'.\n      ')