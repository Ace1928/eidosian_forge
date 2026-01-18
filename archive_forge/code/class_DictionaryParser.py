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
class DictionaryParser(object):
    """A helper class to parse elements out of a JSON dictionary."""

    def __init__(self, cls, dictionary):
        """Initializes the parser.

    Args:
      cls: class, The class that is doing the parsing (used for error messages).
      dictionary: dict, The JSON dictionary to parse.
    """
        self.__cls = cls
        self.__dictionary = dictionary
        self.__args = {}

    def Args(self):
        """Gets the dictionary of all parsed arguments.

    Returns:
      dict, The dictionary of field name to value for all parsed arguments.
    """
        return self.__args

    def _Get(self, field, default, required):
        if required and field not in self.__dictionary:
            raise ParseError('Required field [{0}] not found while parsing [{1}]'.format(field, self.__cls))
        return self.__dictionary.get(field, default)

    def Parse(self, field, required=False, default=None, func=None):
        """Parses a single element out of the dictionary.

    Args:
      field: str, The name of the field to parse.
      required: bool, If the field must be present or not (False by default).
      default: str or dict, The value to use if a non-required field is not
        present.
      func: An optional function to call with the value before returning (if
        value is not None).  It takes a single parameter and returns a single
        new value to be used instead.

    Raises:
      ParseError: If a required field is not found or if the field parsed is a
        list.
    """
        value = self._Get(field, default, required)
        if value is not None:
            if isinstance(value, list):
                raise ParseError('Did not expect a list for field [{field}] in component [{component}]'.format(field=field, component=self.__cls))
            if func:
                value = func(value)
        self.__args[field] = value

    def ParseList(self, field, required=False, default=None, func=None, sort=False):
        """Parses a element out of the dictionary that is a list of items.

    Args:
      field: str, The name of the field to parse.
      required: bool, If the field must be present or not (False by default).
      default: str or dict, The value to use if a non-required field is not
        present.
      func: An optional function to call with each value in the parsed list
        before returning (if the list is not None).  It takes a single parameter
        and returns a single new value to be used instead.
      sort: bool, sort parsed list when it represents an unordered set.

    Raises:
      ParseError: If a required field is not found or if the field parsed is
        not a list.
    """
        value = self._Get(field, default, required)
        if value:
            if not isinstance(value, list):
                raise ParseError('Expected a list for field [{0}] in component [{1}]'.format(field, self.__cls))
            if func:
                value = [func(v) for v in value]
        self.__args[field] = sorted(value) if sort else value

    def ParseDict(self, field, required=False, default=None, func=None):
        """Parses a element out of the dictionary that is a dictionary of items.

    Most elements are dictionaries but the difference between this and the
    normal Parse method is that Parse interprets the value as an object.  Here,
    the value of the element is a dictionary of key:object where the keys are
    unknown.

    Args:
      field: str, The name of the field to parse.
      required: bool, If the field must be present or not (False by default).
      default: str or dict, The value to use if a non-required field is not
        present.
      func: An optional function to call with each value in the parsed dict
        before returning (if the dict is not empty).  It takes a single
        parameter and returns a single new value to be used instead.

    Raises:
      ParseError: If a required field is not found or if the field parsed is
        not a dict.
    """
        value = self._Get(field, default, required)
        if value:
            if not isinstance(value, dict):
                raise ParseError('Expected a dict for field [{0}] in component [{1}]'.format(field, self.__cls))
            if func:
                value = dict(((k, func(v)) for k, v in six.iteritems(value)))
        self.__args[field] = value