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
def _set_is_retired_flag_func(self, is_retired_flag_func):
    """Sets a function for checking retired flags.

    Do not use it. This is a private absl API used to check retired flags
    registered by the absl C++ flags library.

    Args:
      is_retired_flag_func: Callable(str) -> (bool, bool), a function takes flag
        name as parameter, returns a tuple (is_retired, type_is_bool).
    """
    self.__dict__['__is_retired_flag_func'] = is_retired_flag_func