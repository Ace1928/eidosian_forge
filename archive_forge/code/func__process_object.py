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
def _process_object(self, context, objprim):
    try:
        return self.OBJ_BASE_CLASS.obj_from_primitive(objprim, context=context)
    except exception.IncompatibleObjectVersion:
        with excutils.save_and_reraise_exception(reraise=False) as ctxt:
            verkey = '%s.version' % self.OBJ_BASE_CLASS.OBJ_SERIAL_NAMESPACE
            objver = objprim[verkey]
            if objver.count('.') == 2:
                objprim[verkey] = '.'.join(objver.split('.')[:2])
                return self._process_object(context, objprim)
            namekey = '%s.name' % self.OBJ_BASE_CLASS.OBJ_SERIAL_NAMESPACE
            objname = objprim[namekey]
            supported = VersionedObjectRegistry.obj_classes().get(objname, [])
            if self.OBJ_BASE_CLASS.indirection_api and supported:
                return self._do_backport(context, objprim, supported[0])
            else:
                ctxt.reraise = True