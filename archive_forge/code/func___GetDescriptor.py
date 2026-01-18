import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __GetDescriptor(self, name):
    return self.__GetDescriptorByName(self.__ComputeFullName(name))