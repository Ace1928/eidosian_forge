import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __AddImport(self, new_import):
    if new_import not in self.__file_descriptor.additional_imports:
        self.__file_descriptor.additional_imports.append(new_import)