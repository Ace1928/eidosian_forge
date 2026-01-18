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
def _obj_relationship_for(self, field, target_version):
    if not hasattr(self, '_obj_version_manifest') or self._obj_version_manifest is None:
        try:
            return self.obj_relationships[field]
        except KeyError:
            raise exception.ObjectActionError(action='obj_make_compatible', reason='No rule for %s' % field)
    objname = self.fields[field].objname
    if objname not in self._obj_version_manifest:
        return
    return [(target_version, self._obj_version_manifest[objname])]