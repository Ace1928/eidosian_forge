from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def GetZoneOrRegion(args, ignore_property=False, required=True, is_autopilot=False):
    """Get a location (zone or region) from argument or property.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.
    ignore_property: bool, if true, will get location only from argument.
    required: bool, if true, lack of zone will cause raise an exception.
    is_autopilot: bool, if true, region property will take precedence over zone.

  Raises:
    MinimumArgumentException: if location if required and not provided.

  Returns:
    str, a location selected by user.
  """
    location = getattr(args, 'location', None)
    zone = getattr(args, 'zone', None)
    region = getattr(args, 'region', None)
    if ignore_property:
        location_property = None
    elif is_autopilot and properties.VALUES.compute.region.Get():
        location_property = properties.VALUES.compute.region.Get()
    elif properties.VALUES.compute.zone.Get():
        location_property = properties.VALUES.compute.zone.Get()
    else:
        location_property = properties.VALUES.compute.region.Get()
    location = location or region or zone or location_property
    if required and (not location):
        raise calliope_exceptions.MinimumArgumentException(['--location', '--zone', '--region'])
    return location