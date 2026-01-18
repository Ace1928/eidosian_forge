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
class _ConfigurationMetaClass(type):
    """The metaclass for all Configuration types.

  This class is needed to store a class specific list of all ConfigOptions in
  cls._options, and insert a __slots__ variable into the class dict before the
  class is created to impose immutability.
  """

    def __new__(metaclass, classname, bases, classDict):
        if classname == '_MergedConfiguration':
            return type.__new__(metaclass, classname, bases, classDict)
        if object in bases:
            classDict['__slots__'] = ['_values']
        else:
            classDict['__slots__'] = []
        cls = type.__new__(metaclass, classname, bases, classDict)
        if object not in bases:
            options = {}
            for c in reversed(cls.__mro__):
                if '_options' in c.__dict__:
                    options.update(c.__dict__['_options'])
            cls._options = options
            for option, value in cls.__dict__.items():
                if isinstance(value, ConfigOption):
                    if option in cls._options:
                        raise TypeError('%s cannot be overridden (%s)' % (option, cls.__name__))
                    cls._options[option] = value
                    value._cls = cls
        return cls