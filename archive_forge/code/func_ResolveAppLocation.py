from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client as app_engine_api
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def ResolveAppLocation(project_ref, locations_client=None):
    """Gets the default location from the Cloud Tasks API.

  If an AppEngine app exists, the default location is the location where the
  app exists.

  Args:
    project_ref: The project resource to look up the location for.
    locations_client: The project resource used to look up locations.

  Returns:
    The location. Some examples: 'us-central1', 'us-east4'

  Raises:
    RegionResolvingError: If we are unable to determine a default location
      for the given project.
  """
    if not locations_client:
        locations_client = GetApiAdapter(calliope_base.ReleaseTrack.GA).locations
    locations = list(locations_client.List(project_ref))
    if len(locations) >= 1 and AppEngineAppExists():
        location = locations[0].labels.additionalProperties[0].value
        if len(locations) > 1:
            log.warning(constants.APP_ENGINE_DEFAULT_LOCATION_WARNING.format(location))
        return location
    raise RegionResolvingError('Please use the location flag to manually specify a location.')