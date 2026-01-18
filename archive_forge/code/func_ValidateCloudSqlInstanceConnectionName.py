from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateCloudSqlInstanceConnectionName(connection_name):
    """Validates the connection name of a CloudSQL instance, must be in the form '{project_id}:{region}:{instance_id}'.

  Args:
    connection_name: The CloudSQL instance connection name string.

  Returns:
    The connection name string.
  Raises:
    BadArgumentException: when the input string does not match the pattern.
  """
    pattern = re.compile('^([^:]+:){2}[^:]+$')
    if not pattern.match(connection_name):
        raise exceptions.BadArgumentException('--instance-connection-name', 'The instance connection name should be in the format project_id:region:instance_id')
    return connection_name