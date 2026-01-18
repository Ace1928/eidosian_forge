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
class BaseConfiguration(six_subset.with_metaclass(_ConfigurationMetaClass, object)):
    """A base class for a configuration object.

  Subclasses should provide validation functions for every configuration option
  they accept. Any public function decorated with ConfigOption is assumed to be
  a validation function for an option of the same name. All validation functions
  take a single non-None value to validate and must throw an exception or return
  the value to store.

  This class forces subclasses to be immutable and exposes a read-only
  property for every accepted configuration option. Configuration options set by
  passing keyword arguments to the constructor. The constructor and merge
  function are designed to avoid creating redundant copies and may return
  the configuration objects passed to them if appropriate.

  Setting an option to None is the same as not specifying the option except in
  the case where the 'config' argument is given. In this case the value on
  'config' of the same name is ignored. Options that are not specified will
  return 'None' when accessed.
  """
    _options = {}

    def __new__(cls, config=None, **kwargs):
        """Immutable constructor.

    If 'config' is non-None all configuration options will default to the value
    it contains unless the configuration option is explicitly set to 'None' in
    the keyword arguments. If 'config' is None then all configuration options
    default to None.

    Args:
      config: Optional base configuration providing default values for
        parameters not specified in the keyword arguments.
      **kwargs: Configuration options to store on this object.

    Returns:
      Either a new Configuration object or (if it would be equivalent)
      the config argument unchanged, but never None.
    """
        if config is None:
            pass
        elif isinstance(config, BaseConfiguration):
            if cls is config.__class__ and config.__is_stronger(**kwargs):
                return config
            for key, value in config._values.items():
                if issubclass(cls, config._options[key]._cls):
                    kwargs.setdefault(key, value)
        else:
            raise datastore_errors.BadArgumentError('config argument should be Configuration (%r)' % (config,))
        obj = super(BaseConfiguration, cls).__new__(cls)
        obj._values = {}
        for key, value in kwargs.items():
            if value is not None:
                try:
                    config_option = obj._options[key]
                except KeyError as err:
                    raise TypeError('Unknown configuration option (%s)' % err)
                value = config_option.validator(value)
                if value is not None:
                    obj._values[key] = value
        return obj

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, BaseConfiguration):
            return NotImplemented
        return self._options == other._options and self._values == other._values

    def __ne__(self, other):
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return equal
        return not equal

    def __hash__(self):
        return hash(frozenset(self._values.items())) ^ hash(frozenset(self._options.items()))

    def __repr__(self):
        args = []
        for key_value in sorted(self._values.items()):
            args.append('%s=%r' % key_value)
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))

    def __is_stronger(self, **kwargs):
        """Internal helper to ask whether a configuration is stronger than another.

    A configuration is stronger when it contains every name/value pair in
    kwargs.

    Example: a configuration with:
      (deadline=5, on_configuration=None, read_policy=EVENTUAL_CONSISTENCY)
    is stronger than:
      (deadline=5, on_configuration=None)
    but not stronger than:
      (deadline=5, on_configuration=None, read_policy=None)
    or
      (deadline=10, on_configuration=None, read_policy=None).

    More formally:
      - Any value is stronger than an unset value;
      - Any value is stronger than itself.

    Returns:
      True if each of the self attributes is stronger than the
    corresponding argument.
    """
        for key, value in kwargs.items():
            if key not in self._values or value != self._values[key]:
                return False
        return True

    @classmethod
    def is_configuration(cls, obj):
        """True if configuration obj handles all options of this class.

    Use this method rather than isinstance(obj, cls) to test if a
    configuration object handles the options of cls (is_configuration
    is handled specially for results of merge which may handle the options
    of unrelated configuration classes).

    Args:
      obj: the object to test.
    """
        return isinstance(obj, BaseConfiguration) and obj._is_configuration(cls)

    def _is_configuration(self, cls):
        return isinstance(self, cls)

    def merge(self, config):
        """Merge two configurations.

    The configuration given as an argument (if any) takes priority;
    defaults are filled in from the current configuration.

    Args:
      config: Configuration providing overrides, or None (but cannot
        be omitted).

    Returns:
      Either a new configuration object or (if it would be equivalent)
      self or the config argument unchanged, but never None.

    Raises:
      BadArgumentError if self or config are of configurations classes
      with conflicting options (i.e. the same option name defined in
      two different configuration classes).
    """
        if config is None or config is self:
            return self
        if not (isinstance(config, _MergedConfiguration) or isinstance(self, _MergedConfiguration)):
            if isinstance(config, self.__class__):
                for key in self._values:
                    if key not in config._values:
                        break
                else:
                    return config
            if isinstance(self, config.__class__):
                if self.__is_stronger(**config._values):
                    return self

            def _quick_merge(obj):
                obj._values = self._values.copy()
                obj._values.update(config._values)
                return obj
            if isinstance(config, self.__class__):
                return _quick_merge(type(config)())
            if isinstance(self, config.__class__):
                return _quick_merge(type(self)())
        return _MergedConfiguration(config, self)

    def __getstate__(self):
        return {'_values': self._values}

    def __setstate__(self, state):
        obj = self.__class__(**state['_values'])
        self._values = obj._values