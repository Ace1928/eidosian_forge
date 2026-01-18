from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
class DictionaryWriter(object):
    """Class to help writing these objects back out to a dictionary."""

    def __init__(self, obj):
        self.__obj = obj
        self.__dictionary = {}

    @staticmethod
    def AttributeGetter(attrib):

        def Inner(obj):
            if obj is None:
                return None
            return getattr(obj, attrib)
        return Inner

    def Write(self, field, func=None):
        """Writes the given field to the dictionary.

    This gets the value of the attribute named field from self, and writes that
    to the dictionary.  The field is not written if the value is not set.

    Args:
      field: str, The field name.
      func: An optional function to call on the value of the field before
        writing it to the dictionary.
    """
        value = getattr(self.__obj, field)
        if value is None:
            return
        if func:
            value = func(value)
        self.__dictionary[field] = value

    def WriteList(self, field, func=None):
        """Writes the given list field to the dictionary.

    This gets the value of the attribute named field from self, and writes that
    to the dictionary.  The field is not written if the value is not set.

    Args:
      field: str, The field name.
      func: An optional function to call on each value in the list before
        writing it to the dictionary.
    """
        list_func = None
        if func:

            def ListMapper(values):
                return [func(v) for v in values]
            list_func = ListMapper
        self.Write(field, func=list_func)

    def WriteDict(self, field, func=None):
        """Writes the given dict field to the dictionary.

    This gets the value of the attribute named field from self, and writes that
    to the dictionary.  The field is not written if the value is not set.

    Args:
      field: str, The field name.
      func: An optional function to call on each value in the dict before
        writing it to the dictionary.
    """

        def DictMapper(values):
            return dict(((k, func(v)) for k, v in six.iteritems(values)))
        dict_func = DictMapper if func else None
        self.Write(field, func=dict_func)

    def Dictionary(self):
        return self.__dictionary