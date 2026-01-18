from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def BuildFunctionDestinationMessage(self, project_id, destination_function, destination_function_location):
    """Builds a Function Destination message with the given data.

    Args:
      project_id: the ID of the project.
      destination_function: str or None, the Trigger's destination Function ID.
      destination_function_location: str or None, the location of the Trigger's
        destination Function. It defaults to the Trigger's location.

    Returns:
      A Destination message with a Function destination.
    """
    function_message = 'projects/{}/locations/{}/functions/{}'.format(project_id, destination_function_location, destination_function)
    return self._messages.Destination(cloudFunction=function_message)