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
def ActiveDirectoryConfig(sql_messages, domain=None):
    """Generates the Active Directory configuration for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    domain: string, the Active Directory domain value.

  Returns:
    sql_messages.SqlActiveDirectoryConfig object.
  """
    config = sql_messages.SqlActiveDirectoryConfig(domain=domain)
    return config