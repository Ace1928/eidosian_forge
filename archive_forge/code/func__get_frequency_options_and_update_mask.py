from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def _get_frequency_options_and_update_mask(self, start_date, end_date, frequency):
    """Returns a tuple of messages.FrequencyOptions and update_mask list."""
    update_mask = []
    if start_date is not None:
        start_date_message = self.messages.Date(year=start_date.year, month=start_date.month, day=start_date.day)
        update_mask.append('frequencyOptions.startDate')
    else:
        start_date_message = None
    if end_date is not None:
        end_date_message = self.messages.Date(year=end_date.year, month=end_date.month, day=end_date.day)
        update_mask.append('frequencyOptions.endDate')
    else:
        end_date_message = None
    if frequency is not None:
        frequency_message = getattr(self.messages.FrequencyOptions.FrequencyValueValuesEnum, frequency.upper())
        update_mask.append('frequencyOptions.frequency')
    else:
        frequency_message = None
    return (self.messages.FrequencyOptions(startDate=start_date_message, endDate=end_date_message, frequency=frequency_message), update_mask)