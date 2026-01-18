from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def AddDestGroupCreateIpAndFqdnArgs(parser):
    """Adds IP and FQDN args for DestGroup Create command.

  Args:
    parser: An argparse.ArgumentParser-like object. It is mocked out in order to
      capture some information, but behaves like an ArgumentParser.
  """
    parser.add_argument('--ip-range-list', help='List of ip-ranges in the Destination Group.')
    parser.add_argument('--fqdn-list', help='List of FQDNs in the Destination Group.')