from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import instances
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
def _GetUriFromResource(resource):
    """Returns the URI for resource."""
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    return client.resource_parser.Create('sql.instances', project=resource.project, instance=resource.name).SelfLink()