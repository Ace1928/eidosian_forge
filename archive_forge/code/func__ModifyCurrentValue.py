from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def _ModifyCurrentValue(remove, add, clear, set_, current_value_thunk):
    """Performs the modification of the current value based on the args.

  Args:
    remove: list[str], items to be removed from the current value.
    add: list[str], items to be added to the current value.
    clear: bool, whether or not to clear the current value.
    set_: list[str], items to replace the current value.
    current_value_thunk: zero-arg function that returns the current value of the
      attribute to be updated. Will be called lazily if required.

  Raises:
    ValueError: if more than one arg is set.

  Returns:
    List of str: the new value for the field, or None if no change is required.
  """
    if sum(map(bool, (remove, add, clear, set_))) > 1:
        raise ValueError('At most one arg can be set.')
    if remove is not None:
        current_value = current_value_thunk()
        new_value = [x for x in current_value if x not in remove]
    elif add is not None:
        current_value = current_value_thunk()
        new_value = current_value + [x for x in add if x not in current_value]
    elif clear:
        return []
    elif set_ is not None:
        return set_
    else:
        return None
    if new_value != current_value:
        return new_value
    else:
        return None