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
def AddCloudRunConfigFlag(parser, suppressed=False):
    """Adds a --cloud-run-config flag to parser."""
    help_text = 'Configurations for Cloud Run addon, requires `--addons=CloudRun` for create\nand `--update-addons=CloudRun=ENABLED` for update.\n\n*load-balancer-type*::: (Optional) Type of load-balancer-type EXTERNAL or\nINTERNAL.\n\nExamples:\n\n  $ {command} example-cluster --cloud-run-config=load-balancer-type=INTERNAL\n'
    _AddLegacyCloudRunFlag(parser, '--{0}-config', metavar='load-balancer-type=EXTERNAL', type=arg_parsers.ArgDict(spec={'load-balancer-type': lambda x: x.upper()}), help=help_text, hidden=suppressed)