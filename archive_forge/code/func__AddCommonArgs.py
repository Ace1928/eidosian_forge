from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from six.moves import input  # pylint: disable=redefined-builtin
def _AddCommonArgs(parser):
    """Register common flags for this command.

  Args:
    parser: An argparse.ArgumentParser-like object. It is mocked out in order to
      capture some information, but behaves like an ArgumentParser.
  """
    parser.add_argument('name', metavar='NAME', help='The name of the cluster to update.')
    parser.add_argument('--node-pool', help='Node pool to be updated.')
    parser.add_argument('--timeout', type=int, default=3600, hidden=True, help='Timeout (seconds) for waiting on the operation to complete.')
    flags.AddAsyncFlag(parser)