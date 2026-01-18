from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
class BuiltinHandler(validation.Validated):
    """Class representing built-in handler directives in application info.

  This class permits arbitrary keys, but their values must be described by the
  `validation.Options` object that is returned by `ATTRIBUTES`.
  """

    class DynamicAttributes(dict):
        """Provides a dictionary object that will always claim to have a key.

    This dictionary returns a fixed value for any `get` operation. The fixed
    value that you pass in as a constructor parameter should be a
    `validation.Validated` object.
    """

        def __init__(self, return_value, **parameters):
            self.__return_value = return_value
            dict.__init__(self, parameters)

        def __contains__(self, _):
            return True

        def __getitem__(self, _):
            return self.__return_value
    ATTRIBUTES = DynamicAttributes(validation.Optional(validation.Options((ON, ON_ALIASES), (OFF, OFF_ALIASES))))

    def __init__(self, **attributes):
        """Ensures all BuiltinHandler objects at least use the `default` attribute.

    Args:
      **attributes: The attributes that you want to use.
    """
        self.builtin_name = ''
        super(BuiltinHandler, self).__init__(**attributes)

    def __setattr__(self, key, value):
        """Allows `ATTRIBUTES.iteritems()` to return set of items that have values.

    Whenever `validate` calls `iteritems()`, it is always called on
    `ATTRIBUTES`, not on `__dict__`, so this override is important to ensure
    that functions such as `ToYAML()` return the correct set of keys.

    Args:
      key: The key for the `iteritem` that you want to set.
      value: The value for the `iteritem` that you want to set.

    Raises:
      MultipleBuiltinsSpecified: If more than one built-in is defined in a list
          element.
    """
        if key == 'builtin_name':
            object.__setattr__(self, key, value)
        elif not self.builtin_name:
            self.ATTRIBUTES[key] = ''
            self.builtin_name = key
            super(BuiltinHandler, self).__setattr__(key, value)
        else:
            raise appinfo_errors.MultipleBuiltinsSpecified('More than one builtin defined in list element.  Each new builtin should be prefixed by "-".')

    def __getattr__(self, key):
        if key.startswith('_'):
            raise AttributeError
        return None

    def GetUnnormalized(self, key):
        try:
            return super(BuiltinHandler, self).GetUnnormalized(key)
        except AttributeError:
            return getattr(self, key)

    def ToDict(self):
        """Converts a `BuiltinHander` object to a dictionary.

    Returns:
      A dictionary in `{builtin_handler_name: on/off}` form
    """
        return {self.builtin_name: getattr(self, self.builtin_name)}

    @classmethod
    def IsDefined(cls, builtins_list, builtin_name):
        """Finds if a builtin is defined in a given list of builtin handler objects.

    Args:
      builtins_list: A list of `BuiltinHandler` objects, typically
          `yaml.builtins`.
      builtin_name: The name of the built-in that you want to determine whether
          it is defined.

    Returns:
      `True` if `builtin_name` is defined by a member of `builtins_list`; all
      other results return `False`.
    """
        for b in builtins_list:
            if b.builtin_name == builtin_name:
                return True
        return False

    @classmethod
    def ListToTuples(cls, builtins_list):
        """Converts a list of `BuiltinHandler` objects.

    Args:
      builtins_list: A list of `BuildinHandler` objects to convert to tuples.

    Returns:
      A list of `(name, status)` that is derived from the `BuiltinHandler`
      objects.
    """
        return [(b.builtin_name, getattr(b, b.builtin_name)) for b in builtins_list]

    @classmethod
    def Validate(cls, builtins_list, runtime=None):
        """Verifies that all `BuiltinHandler` objects are valid and not repeated.

    Args:
      builtins_list: A list of `BuiltinHandler` objects to validate.
      runtime: If you specify this argument, warnings are generated for
          built-ins that have been deprecated in the given runtime.

    Raises:
      InvalidBuiltinFormat: If the name of a `BuiltinHandler` object cannot be
          determined.
      DuplicateBuiltinsSpecified: If a `BuiltinHandler` name is used more than
          once in the list.
    """
        seen = set()
        for b in builtins_list:
            if not b.builtin_name:
                raise appinfo_errors.InvalidBuiltinFormat('Name of builtin for list object %s could not be determined.' % b)
            if b.builtin_name in seen:
                raise appinfo_errors.DuplicateBuiltinsSpecified('Builtin %s was specified more than once in one yaml file.' % b.builtin_name)
            if b.builtin_name == 'datastore_admin' and runtime == 'python':
                logging.warning('The datastore_admin builtin is deprecated. You can find information on how to enable it through the Administrative Console here: http://developers.google.com/appengine/docs/adminconsole/datastoreadmin.html')
            elif b.builtin_name == 'mapreduce' and runtime == 'python':
                logging.warning('The mapreduce builtin is deprecated. You can find more information on how to configure and use it here: http://developers.google.com/appengine/docs/python/dataprocessing/overview.html')
            seen.add(b.builtin_name)