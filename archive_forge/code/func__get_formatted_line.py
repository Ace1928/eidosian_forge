from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import datetime
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
import six
def _get_formatted_line(display_name, value, default_value=None):
    """Returns a formatted line for ls -L output."""
    if value is not None:
        if value and (isinstance(value, dict) or isinstance(value, list)):
            return resource_util.get_metadata_json_section_string(display_name, value)
        elif isinstance(value, datetime.datetime):
            return resource_util.get_padded_metadata_time_line(display_name, value)
        elif isinstance(value, errors.CloudApiError):
            return resource_util.get_padded_metadata_key_value_line(display_name, str(value))
        return resource_util.get_padded_metadata_key_value_line(display_name, value)
    elif default_value is not None:
        return resource_util.get_padded_metadata_key_value_line(display_name, default_value)
    return None