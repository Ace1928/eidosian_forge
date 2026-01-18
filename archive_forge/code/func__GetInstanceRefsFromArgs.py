from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GetInstanceRefsFromArgs(args, client):
    """Get validated refs to source and destination instances from args."""
    validate.ValidateInstanceName(args.source)
    validate.ValidateInstanceName(args.destination)
    source_instance_ref = client.resource_parser.Parse(args.source, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    destination_instance_ref = client.resource_parser.Parse(args.destination, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    _CheckSourceAndDestination(source_instance_ref, destination_instance_ref)
    return (source_instance_ref, destination_instance_ref)