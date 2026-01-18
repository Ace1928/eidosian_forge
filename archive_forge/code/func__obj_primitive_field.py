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
@classmethod
def _obj_primitive_field(cls, primitive, field, default=obj_fields.UnspecifiedDefault):
    key = cls._obj_primitive_key(field)
    if default == obj_fields.UnspecifiedDefault:
        return primitive[key]
    else:
        return primitive.get(key, default)