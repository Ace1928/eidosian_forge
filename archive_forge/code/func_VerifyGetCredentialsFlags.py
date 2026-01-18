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
def VerifyGetCredentialsFlags(args):
    """Verifies that the passed flags are valid for get-credentials.

  Only one of the following flags may be specified at a time:
  --cross-connect, --private-endpoint-fqdn, or --internal-ip

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Raises:
    util.Error, if flags conflict.
  """
    if args.IsSpecified('internal_ip') + args.IsSpecified('cross_connect_subnetwork') + args.IsSpecified('private_endpoint_fqdn') > 1:
        raise util.Error(constants.CONFLICTING_GET_CREDS_FLAGS_ERROR_MSG)