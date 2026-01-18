from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config.troubleshoot import utils
from googlecloudsdk.core import exceptions
def _GetValueFromUri(uri, field):
    """Gets the value of the desired field from the provided uri.

  The uri should be an array containing the field keys directly followed by
  their values. An example array is [projects, example-project], where
  `projects` is the field and `example-project` is its value.

  Args:
    uri: the uri from which to get fields, in array form.
    field: the desired field to Get

  Returns:
    The value of the field in the uri, None if the field doesn't exist.
  """
    index = uri.index(field)
    if index == -1:
        return None
    return uri[index + 1]