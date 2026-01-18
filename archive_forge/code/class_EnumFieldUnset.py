import functools
import inspect
import logging
from oslo_config import cfg
from oslo_utils import excutils
import webob.exc
from oslo_versionedobjects._i18n import _
class EnumFieldUnset(VersionedObjectsException):
    msg_fmt = _('%(fieldname)s missing field type')