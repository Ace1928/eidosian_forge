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
def _CheckSourceAndDestination(source_instance_ref, destination_instance_ref):
    """Verify that the source and destination instance ids are different."""
    if source_instance_ref.project != destination_instance_ref.project:
        raise exceptions.ArgumentError('The source and the clone instance must belong to the same project: "{src}" != "{dest}".'.format(src=source_instance_ref.project, dest=destination_instance_ref.project))