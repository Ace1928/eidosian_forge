from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def CreateReservationName(api_version):
    """Creates the target reservation name from args.

  Args:
    api_version: The api version (e.g. v2 or v2alpha1)

  Returns:
    Handler which sets request.queuedResource.reservationName
  """

    def Process(ref, args, request):
        del ref
        if args.reservation_host_project and args.reservation_host_folder or (args.reservation_host_folder and args.reservation_host_organization) or (args.reservation_host_organization and args.reservation_host_project):
            raise exceptions.ConflictingArgumentsException('Only one reservation host is permitted')
        pattern = '{}/{}/locations/{}/reservations/-'
        reservation_name = None
        if args.reservation_host_project:
            reservation_name = pattern.format('projects', args.reservation_host_project, args.zone)
        elif args.reservation_host_folder:
            reservation_name = pattern.format('folders', args.reservation_host_folder, args.zone)
        elif args.reservation_host_organization:
            reservation_name = pattern.format('organizations', args.reservation_host_organization, args.zone)
        elif api_version == 'v2' and hasattr(args, 'reserved') and args.reserved:
            project = properties.VALUES.core.project.GetOrFail()
            reservation_name = pattern.format('projects', project, args.zone)
        if reservation_name:
            request.queuedResource.reservationName = reservation_name
        return request
    return Process