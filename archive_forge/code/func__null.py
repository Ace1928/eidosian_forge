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
def _null(self, obj, attr):
    if self.nullable:
        return None
    elif self._default != UnspecifiedDefault:
        return self._type.coerce(obj, attr, copy.deepcopy(self._default))
    else:
        raise ValueError(_("Field `%s' cannot be None") % attr)