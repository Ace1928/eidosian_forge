from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def AddTopicReservationResource(resource_ref, args, request):
    """Returns an updated `request` with a resource path on the reservation."""
    del resource_ref, args
    topic = request.topic
    if not _HasReservation(topic):
        return request
    resource, _ = GetResourceInfo(request)
    project = DeriveProjectFromResource(resource)
    region = DeriveRegionFromLocation(DeriveLocationFromResource(resource))
    reservation = topic.reservationConfig.throughputReservation
    request.topic.reservationConfig.throughputReservation = '{}{}/{}{}/{}{}'.format(PROJECTS_RESOURCE_PATH, project, LOCATIONS_RESOURCE_PATH, region, RESERVATIONS_RESOURCE_PATH, reservation)
    return request