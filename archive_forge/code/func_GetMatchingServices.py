from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import text
import six
def GetMatchingServices(all_services, args_services):
    """Return a list of services to act on based on user arguments.

  Args:
    all_services: list of Services representing all services in the project.
    args_services: list of string, service IDs to filter for, from arguments
      given by the user to the command line. If empty, match all services.

  Returns:
    list of matching Services sorted by the order they were given to the
      command line.

  Raises:
    ServiceValidationError: If an improper combination of arguments is given
  """
    if not args_services:
        args_services = sorted((s.id for s in all_services))
    else:
        _ValidateServicesAreSubset(args_services, [s.id for s in all_services])
    matching_services = []
    for service_id in args_services:
        matching_services += [s for s in all_services if s.id == service_id]
    return matching_services