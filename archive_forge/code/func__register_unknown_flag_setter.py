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
def _register_unknown_flag_setter(self, setter):
    """Allow set default values for undefined flags.

    Args:
      setter: Method(name, value) to call to __setattr__ an unknown flag. Must
        raise NameError or ValueError for invalid name/value.
    """
    self.__dict__['__set_unknown'] = setter