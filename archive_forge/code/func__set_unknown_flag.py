from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _set_unknown_flag(self, name, value):
    """Returns value if setting flag |name| to |value| returned True.

    Args:
      name: str, name of the flag to set.
      value: Value to set.

    Returns:
      Flag value on successful call.

    Raises:
      UnrecognizedFlagError
      IllegalFlagValueError
    """
    setter = self.__dict__['__set_unknown']
    if setter:
        try:
            setter(name, value)
            return value
        except (TypeError, ValueError):
            raise _exceptions.IllegalFlagValueError('"{1}" is not valid for --{0}'.format(name, value))
        except NameError:
            pass
    raise _exceptions.UnrecognizedFlagError(name, value)