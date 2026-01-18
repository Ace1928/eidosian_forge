from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def _add_single_transfer_prefix(prefix_to_check, prefix_to_add, resource_string):
    """Adds prefix to one resource string if necessary."""
    if re.match(prefix_to_check, resource_string):
        return resource_string
    return prefix_to_add + resource_string