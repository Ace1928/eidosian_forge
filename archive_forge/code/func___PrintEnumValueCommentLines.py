import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def __PrintEnumValueCommentLines(self, enum_value):
    if enum_value.description:
        width = self.__printer.CalculateWidth() - 3
        for line in textwrap.wrap(enum_value.description, width):
            self.__printer('// %s', line)