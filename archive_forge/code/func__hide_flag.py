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
def _hide_flag(self, name):
    """Marks the flag --name as hidden."""
    self.__dict__['__hiddenflags'].add(name)