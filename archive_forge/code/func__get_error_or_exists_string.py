from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
def _get_error_or_exists_string(value):
    """Returns error if value is error or existence string."""
    if isinstance(value, errors.XmlApiError):
        return value
    else:
        return resource_util.get_exists_string(value)