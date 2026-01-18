from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ModifyUptimeCheck(uptime_check, messages, args, regions, user_labels, headers, status_classes, status_codes, update=False):
    """Modifies an UptimeCheckConfig based on the args and other inputs.

  Args:
    uptime_check: UptimeCheckConfig that is being modified.
    messages: Object containing information about all message types allowed.
    args: Flags provided by the user.
    regions: Potentially updated selected regions.
    user_labels: Potentially updated user labels.
    headers: Potentially updated HTTP headers.
    status_classes: Potentially updated allowed status classes.
    status_codes: Potentially updated allowed status codes.
    update: If this is an update operation (true) or a create operation (false).

  Returns:
     The updated UptimeCheckConfig object.
  """
    if args.display_name is not None:
        uptime_check.displayName = args.display_name
    if args.timeout is not None:
        uptime_check.timeout = str(args.timeout) + 's'
    if args.period is not None:
        period_mapping = {'1': '60s', '5': '300s', '10': '600s', '15': '900s'}
        uptime_check.period = period_mapping.get(args.period)
    if regions is not None:
        if uptime_check.syntheticMonitor is not None:
            raise calliope_exc.InvalidArgumentException('regions', 'Should not be set or updated for Synthetic Monitor.')
        region_mapping = {'usa-oregon': messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.USA_OREGON, 'usa-iowa': messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.USA_IOWA, 'usa-virginia': messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.USA_VIRGINIA, 'europe': messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.EUROPE, 'south-america': messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.SOUTH_AMERICA, 'asia-pacific': messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.ASIA_PACIFIC}
        uptime_check.selectedRegions = []
        for region in regions:
            uptime_check.selectedRegions.append(region_mapping.get(region))
        uptime_check.userLabels = user_labels
    SetUptimeCheckMatcherFields(args, messages, uptime_check)
    SetUptimeCheckProtocolFields(args, messages, uptime_check, headers, status_classes, status_codes, update)
    return uptime_check