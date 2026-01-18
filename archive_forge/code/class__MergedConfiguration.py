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
class _MergedConfiguration(BaseConfiguration):
    """Helper class to handle merges of configurations.

  Instances of _MergedConfiguration are in some sense "subclasses" of the
  argument configurations, i.e.:
  - they handle exactly the configuration options of the argument configurations
  - the value of these options is taken in priority order from the arguments
  - isinstance is true on this configuration if it is true on any of the
    argument configurations
  This class raises an exception if two argument configurations have an option
  with the same name but coming from a different configuration class.
  """
    __slots__ = ['_values', '_configs', '_options', '_classes']

    def __new__(cls, *configs):
        obj = super(BaseConfiguration, cls).__new__(cls)
        obj._configs = configs
        obj._options = {}
        for config in configs:
            for name, option in config._options.items():
                if name in obj._options:
                    if option is not obj._options[name]:
                        error = "merge conflict on '%s' from '%s' and '%s'" % (name, option._cls.__name__, obj._options[name]._cls.__name__)
                        raise datastore_errors.BadArgumentError(error)
                obj._options[name] = option
        obj._values = {}
        for config in reversed(configs):
            for name, value in config._values.items():
                obj._values[name] = value
        return obj

    def __repr__(self):
        return '%s%r' % (self.__class__.__name__, tuple(self._configs))

    def _is_configuration(self, cls):
        for config in self._configs:
            if config._is_configuration(cls):
                return True
        return False

    def __getattr__(self, name):
        if name in self._options:
            if name in self._values:
                return self._values[name]
            else:
                return None
        raise AttributeError("Configuration has no attribute '%s'" % (name,))

    def __getstate__(self):
        return {'_configs': self._configs}

    def __setstate__(self, state):
        obj = _MergedConfiguration(*state['_configs'])
        self._values = obj._values
        self._configs = obj._configs
        self._options = obj._options