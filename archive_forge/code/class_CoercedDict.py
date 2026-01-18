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
class CoercedDict(CoercedCollectionMixin, dict):
    """Dict which coerces its values

    Dict implementation which overrides all element-adding methods and
    coercing the element(s) being added to the required element type
    """

    def _coerce_dict(self, d):
        res = {}
        for key, element in d.items():
            res[key] = self._coerce_item(key, element)
        return res

    def _coerce_item(self, key, item):
        if not isinstance(key, str):
            raise KeyTypeError(str, key)
        if hasattr(self, '_element_type') and self._element_type is not None:
            att_name = '%s[%s]' % (self._field, key)
            return self._element_type.coerce(self._obj, att_name, item)
        else:
            return item

    def __setitem__(self, key, value):
        super(CoercedDict, self).__setitem__(key, self._coerce_item(key, value))

    def update(self, other=None, **kwargs):
        if other is not None:
            super(CoercedDict, self).update(self._coerce_dict(other), **self._coerce_dict(kwargs))
        else:
            super(CoercedDict, self).update(**self._coerce_dict(kwargs))

    def setdefault(self, key, default=None):
        return super(CoercedDict, self).setdefault(key, self._coerce_item(key, default))