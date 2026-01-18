from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.monitoring import uptime
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.monitoring import flags
from googlecloudsdk.command_lib.monitoring import resource_args
from googlecloudsdk.command_lib.monitoring import util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import log
def ParseSelectedRegions(selected_regions):
    """Convert previously selected regions from enum to flag for update logic."""
    client = uptime.UptimeClient()
    messages = client.messages
    region_mapping = {messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.USA_OREGON: 'usa-oregon', messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.USA_IOWA: 'usa-iowa', messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.USA_VIRGINIA: 'usa-virginia', messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.EUROPE: 'europe', messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.SOUTH_AMERICA: 'south-america', messages.UptimeCheckConfig.SelectedRegionsValueListEntryValuesEnum.ASIA_PACIFIC: 'asia-pacific'}
    return [region_mapping.get(region) for region in selected_regions]