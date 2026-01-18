from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def ApplyMapFlags(old_map, set_flag_value, update_flag_value, clear_flag_value, remove_flag_value, file_flag_value):
    """Determine the new map property from an existing map and parsed arguments.

  Args:
    old_map: the existing map
    set_flag_value: The value from the --set-* flag
    update_flag_value: The value from the --update-* flag
    clear_flag_value: The value from the --clear-* flag
    remove_flag_value: The value from the --remove-* flag
    file_flag_value: The value from the --*-file flag

  Returns:
    A new map with the changes applied.
  """
    if clear_flag_value:
        return {}
    if set_flag_value:
        return set_flag_value
    if file_flag_value:
        return file_flag_value
    if update_flag_value or remove_flag_value:
        old_map = old_map or {}
        remove_flag_value = remove_flag_value or []
        new_map = {k: v for k, v in old_map.items() if k not in remove_flag_value}
        new_map.update(update_flag_value or {})
        return new_map
    return old_map