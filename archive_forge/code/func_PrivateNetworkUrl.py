from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def PrivateNetworkUrl(network):
    """Generates the self-link of the instance's private network.

  Args:
    network: The ID of the network.

  Returns:
    string, the URL of the network.
  """
    client = common_api_util.SqlClient(common_api_util.API_VERSION_DEFAULT)
    network_ref = client.resource_parser.Parse(network, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.networks')
    return network_ref.SelfLink()