from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class ConfigOption(object):
    """A descriptor for a Configuration option.

  This class is used to create a configuration option on a class that inherits
  from BaseConfiguration. A validator function decorated with this class will
  be converted to a read-only descriptor and BaseConfiguration will implement
  constructor and merging logic for that configuration option. A validator
  function takes a single non-None value to validate and either throws
  an exception or returns that value (or an equivalent value). A validator is
  called once at construction time, but only if a non-None value for the
  configuration option is specified the constructor's keyword arguments.
  """

    def __init__(self, validator):
        self.validator = validator

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return obj._values.get(self.validator.__name__, None)

    def __set__(self, obj, value):
        raise AttributeError('Configuration options are immutable (%s)' % (self.validator.__name__,))

    def __call__(self, *args):
        """Gets the first non-None value for this option from the given args.

    Args:
      *arg: Any number of configuration objects or None values.

    Returns:
      The first value for this ConfigOption found in the given configuration
    objects or None.

    Raises:
      datastore_errors.BadArgumentError if a given in object is not a
    configuration object.
    """
        name = self.validator.__name__
        for config in args:
            if isinstance(config, (type(None), apiproxy_stub_map.UserRPC)):
                pass
            elif not isinstance(config, BaseConfiguration):
                raise datastore_errors.BadArgumentError('invalid config argument (%r)' % (config,))
            elif name in config._values and self is config._options[name]:
                return config._values[name]
        return None