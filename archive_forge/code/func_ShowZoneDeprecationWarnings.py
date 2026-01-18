from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def ShowZoneDeprecationWarnings(args):
    """Show warnings if both region and zone are specified or neither is.

  Args:
      args: argparse.Namespace, The arguments that the command was invoked with.
  """
    region_specified = args.IsSpecified('region')
    zone_specified = args.IsSpecified('gce_zone') or args.IsSpecified('zone')
    if not (region_specified or zone_specified):
        log.warning('Starting with release 233.0.0, you will need to specify either a region or a zone to create an instance.')