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
def ConnectToInstance(cmd_args, sql_user):
    """Connects to the instance using the relevant CLI."""
    try:
        log.status.write('Connecting to database with SQL user [{0}].'.format(sql_user))
        execution_utils.Exec(cmd_args)
    except OSError:
        log.error('Failed to execute command "{0}"'.format(' '.join(cmd_args)))
        log.Print(info_holder.InfoHolder())