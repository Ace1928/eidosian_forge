import abc
from collections import abc as collections_abc
import datetime
from distutils import versionpredicate
import re
import uuid
import warnings
import copy
import iso8601
import netaddr
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import _utils
from oslo_versionedobjects import exception
def _coerce_item(self, key, item):
    if not isinstance(key, str):
        raise KeyTypeError(str, key)
    if hasattr(self, '_element_type') and self._element_type is not None:
        att_name = '%s[%s]' % (self._field, key)
        return self._element_type.coerce(self._obj, att_name, item)
    else:
        return item