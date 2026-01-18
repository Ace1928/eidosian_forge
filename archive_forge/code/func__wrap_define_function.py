import logging as _logging
import sys as _sys
from absl.flags import *  # pylint: disable=wildcard-import
from tensorflow.python.util import tf_decorator
def _wrap_define_function(original_function):
    """Wraps absl.flags's define functions so tf.flags accepts old names."""

    def wrapper(*args, **kwargs):
        """Wrapper function that turns old keyword names to new ones."""
        has_old_names = False
        for old_name, new_name in _RENAMED_ARGUMENTS.items():
            if old_name in kwargs:
                has_old_names = True
                value = kwargs.pop(old_name)
                kwargs[new_name] = value
        if has_old_names:
            _logging.warning('Use of the keyword argument names (flag_name, default_value, docstring) is deprecated, please use (name, default, help) instead.')
        return original_function(*args, **kwargs)
    return tf_decorator.make_decorator(original_function, wrapper)