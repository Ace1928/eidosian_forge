from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def ParseAdvertisements(messages, resource_class, args):
    """Parses and validates a completed advertisement configuration from flags.

  Args:
    messages: API messages holder.
    resource_class: RouterBgp or RouterBgpPeer class type to parse for.
    args: Flag arguments to generate configuration from.

  Returns:
    The validated tuple of mode, groups and prefixes.  If mode is DEFAULT,
    validates that no custom advertisements were specified and returns empty
    lists for each.

  Raises:
    CustomWithDefaultError: If custom advertisements were specified at the same
    time as DEFAULT mode.
  """
    mode = None
    if args.advertisement_mode is not None:
        mode = routers_utils.ParseMode(resource_class, args.advertisement_mode)
    groups = None
    if args.set_advertisement_groups is not None:
        groups = routers_utils.ParseGroups(resource_class, args.set_advertisement_groups)
    prefixes = None
    if args.set_advertisement_ranges is not None:
        prefixes = routers_utils.ParseIpRanges(messages, args.set_advertisement_ranges)
    if mode is not None and mode is resource_class.AdvertiseModeValueValuesEnum.DEFAULT:
        if groups or prefixes:
            raise CustomWithDefaultError(messages, resource_class)
        else:
            return (mode, [], [])
    else:
        return (mode, groups, prefixes)