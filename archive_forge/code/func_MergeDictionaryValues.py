from __future__ import absolute_import
import logging
from googlecloudsdk.third_party.appengine.admin.tools.conversion import converters
def MergeDictionaryValues(old_dict, new_dict):
    """Attempts to merge the given dictionaries.

  Warns if a key exists with different values in both dictionaries. In this
  case, the new_dict value trumps the previous value.

  Args:
    old_dict: Existing dictionary.
    new_dict: New dictionary.

  Returns:
    Result of merging the two dictionaries.

  Raises:
    ValueError: If the keys in each dictionary are not unique.
  """
    common_keys = set(old_dict) & set(new_dict)
    if common_keys:
        conflicting_keys = set((key for key in common_keys if old_dict[key] != new_dict[key]))
        if conflicting_keys:

            def FormatKey(key):
                return "'{key}' has conflicting values '{old}' and '{new}'. Using '{new}'.".format(key=key, old=old_dict[key], new=new_dict[key])
            for conflicting_key in conflicting_keys:
                logging.warning(FormatKey(conflicting_key))
    result = old_dict.copy()
    result.update(new_dict)
    return result