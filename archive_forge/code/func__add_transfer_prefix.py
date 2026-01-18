from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def _add_transfer_prefix(prefix_to_check, prefix_to_add, resource_string_or_list):
    """Adds prefix to one resource string or list of strings if necessary."""
    if isinstance(resource_string_or_list, str):
        return _add_single_transfer_prefix(prefix_to_check, prefix_to_add, resource_string_or_list)
    elif isinstance(resource_string_or_list, list):
        return [_add_single_transfer_prefix(prefix_to_check, prefix_to_add, resource_string) for resource_string in resource_string_or_list]
    raise ValueError('Argument must be string or list of strings.')