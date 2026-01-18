import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __AddIfUnknown(self, type_name):
    type_name = self.__names.ClassName(type_name)
    full_type_name = self.__ComputeFullName(type_name)
    if full_type_name not in self.__message_registry.keys() and type_name not in self.__message_registry.keys():
        self.__unknown_types.add(type_name)