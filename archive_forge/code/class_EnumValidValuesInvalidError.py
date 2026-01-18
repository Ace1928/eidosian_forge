import functools
import inspect
import logging
from oslo_config import cfg
from oslo_utils import excutils
import webob.exc
from oslo_versionedobjects._i18n import _
class EnumValidValuesInvalidError(VersionedObjectsException):
    msg_fmt = _('Enum valid values are not valid')