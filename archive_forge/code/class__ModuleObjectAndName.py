from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
class _ModuleObjectAndName(collections.namedtuple('_ModuleObjectAndName', 'module module_name')):
    """Module object and name.

  Fields:
  - module: object, module object.
  - module_name: str, module name.
  """