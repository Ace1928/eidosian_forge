import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __ComputeFullName(self, name):
    return '.'.join(map(six.text_type, self.__current_path[:] + [name]))