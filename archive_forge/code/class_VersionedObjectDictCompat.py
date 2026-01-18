import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
class VersionedObjectDictCompat(object):
    """Mix-in to provide dictionary key access compatibility

    If an object needs to support attribute access using
    dictionary items instead of object attributes, inherit
    from this class. This should only be used as a temporary
    measure until all callers are converted to use modern
    attribute access.
    """

    def __iter__(self):
        for name in self.obj_fields:
            if self.obj_attr_is_set(name) or name in self.obj_extra_fields:
                yield name
    keys = __iter__

    def values(self):
        for name in self:
            yield getattr(self, name)

    def items(self):
        for name in self:
            yield (name, getattr(self, name))

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def get(self, key, value=_NotSpecifiedSentinel):
        if key not in self.obj_fields:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__, key))
        if value != _NotSpecifiedSentinel and (not self.obj_attr_is_set(key)):
            return value
        else:
            return getattr(self, key)

    def update(self, updates):
        for key, value in updates.items():
            setattr(self, key, value)