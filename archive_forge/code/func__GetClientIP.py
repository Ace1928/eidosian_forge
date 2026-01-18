from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances as instances_api_util
from googlecloudsdk.api_lib.sql import network
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.sql import flags as sql_flags
from googlecloudsdk.command_lib.sql import instances as instances_command_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
import six
import six.moves.http_client
def _GetClientIP(instance_ref, sql_client, acl_name):
    """Retrieves given instance and extracts its client ip."""
    instance_info = sql_client.instances.Get(sql_client.MESSAGES_MODULE.SqlInstancesGetRequest(project=instance_ref.project, instance=instance_ref.instance))
    networks = instance_info.settings.ipConfiguration.authorizedNetworks
    client_ip = None
    for net in networks:
        if net.name == acl_name:
            client_ip = net.value
            break
    return (instance_info, client_ip)