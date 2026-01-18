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
def OnPremisesConfiguration(sql_messages, source_ip_address, source_port):
    """Generates the external primary configuration for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    source_ip_address: string, the IP address of the external data source.
    source_port: number, the port number of the external data source.

  Returns:
    sql_messages.OnPremisesConfiguration object.
  """
    return sql_messages.OnPremisesConfiguration(kind='sql#onPremisesConfiguration', hostPort='{0}:{1}'.format(source_ip_address, source_port))