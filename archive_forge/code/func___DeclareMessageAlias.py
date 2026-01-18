import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
def __DeclareMessageAlias(self, schema, alias_for):
    """Declare schema as an alias for alias_for."""
    message = extended_descriptor.ExtendedMessageDescriptor()
    message.name = self.__names.ClassName(schema['id'])
    message.alias_for = alias_for
    self.__DeclareDescriptor(message.name)
    self.__AddImport('from %s import extra_types' % self.__base_files_package)
    self.__RegisterDescriptor(message)