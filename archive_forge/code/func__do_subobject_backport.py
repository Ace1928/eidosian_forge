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
def _do_subobject_backport(to_version, parent, field, primitive):
    obj = getattr(parent, field)
    manifest = hasattr(parent, '_obj_version_manifest') and parent._obj_version_manifest or None
    if isinstance(obj, VersionedObject):
        obj.obj_make_compatible_from_manifest(obj._obj_primitive_field(primitive[field], 'data'), to_version, version_manifest=manifest)
        ver_key = obj._obj_primitive_key('version')
        primitive[field][ver_key] = to_version
    elif isinstance(obj, list):
        for i, element in enumerate(obj):
            element.obj_make_compatible_from_manifest(element._obj_primitive_field(primitive[field][i], 'data'), to_version, version_manifest=manifest)
            ver_key = element._obj_primitive_key('version')
            primitive[field][i][ver_key] = to_version