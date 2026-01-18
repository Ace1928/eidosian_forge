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
def ExecuteCreateSubscriptionRequest(resource_ref, args):
    """Issues a CreateSubscriptionRequest and potentially other requests.

  Args:
    resource_ref: resources.Resource, the resource reference for the resource
      being operated on.
    args: argparse.Namespace, the parsed commandline arguments.

  Returns:
    The created Pub/Sub Lite Subscription.
  """
    psl = PubsubLiteMessages()
    location = GetLocation(args)
    project_id = GetProject(args)
    project_number = six.text_type(ProjectIdToProjectNumber(project_id))
    requires_seek = args.publish_time or args.event_time
    create_request = psl.PubsubliteAdminProjectsLocationsSubscriptionsCreateRequest(parent='{}{}/{}{}'.format(PROJECTS_RESOURCE_PATH, project_number, LOCATIONS_RESOURCE_PATH, location), subscription=psl.Subscription(topic=args.topic, deliveryConfig=psl.DeliveryConfig(deliveryRequirement=GetDeliveryRequirement(args, psl)), exportConfig=GetExportConfig(args, psl, project_number, location, requires_seek)), subscriptionId=args.subscription)
    OverrideEndpointWithRegion(create_request)
    AddSubscriptionTopicResource(resource_ref, args, create_request)
    if not requires_seek:
        UpdateSkipBacklogField(resource_ref, args, create_request)
    client = PubsubLiteClient()
    response = client.admin_projects_locations_subscriptions.Create(create_request)
    if requires_seek:
        seek_request = psl.PubsubliteAdminProjectsLocationsSubscriptionsSeekRequest(name=response.name, seekSubscriptionRequest=GetSeekRequest(args, psl))
        client.admin_projects_locations_subscriptions.Seek(seek_request)
    if requires_seek and create_request.subscription.exportConfig and (args.export_desired_state == 'active'):
        update_request = psl.PubsubliteAdminProjectsLocationsSubscriptionsPatchRequest(name=response.name, updateMask='export_config.desired_state', subscription=psl.Subscription(exportConfig=psl.ExportConfig(desiredState=psl.ExportConfig.DesiredStateValueValuesEnum.ACTIVE)))
        response = client.admin_projects_locations_subscriptions.Patch(update_request)
    return response