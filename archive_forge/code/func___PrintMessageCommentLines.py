import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def __PrintMessageCommentLines(self, message_type):
    """Print the description of this message."""
    description = message_type.description or '%s message type.' % message_type.name
    width = self.__printer.CalculateWidth() - 3
    for line in textwrap.wrap(description, width):
        self.__printer('// %s', line)
    PrintIndentedDescriptions(self.__printer, message_type.enum_types, 'Enums', prefix='// ')
    PrintIndentedDescriptions(self.__printer, message_type.message_types, 'Messages', prefix='// ')
    PrintIndentedDescriptions(self.__printer, message_type.fields, 'Fields', prefix='// ')