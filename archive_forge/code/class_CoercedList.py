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
class CoercedList(CoercedCollectionMixin, list):
    """List which coerces its elements

    List implementation which overrides all element-adding methods and
    coercing the element(s) being added to the required element type
    """

    def _coerce_item(self, index, item):
        if hasattr(self, '_element_type') and self._element_type is not None:
            att_name = '%s[%i]' % (self._field, index)
            return self._element_type.coerce(self._obj, att_name, item)
        else:
            return item

    def __setitem__(self, i, y):
        if type(i) is slice:
            start = i.start or 0
            step = i.step or 1
            coerced_items = [self._coerce_item(start + index * step, item) for index, item in enumerate(y)]
            super(CoercedList, self).__setitem__(i, coerced_items)
        else:
            super(CoercedList, self).__setitem__(i, self._coerce_item(i, y))

    def append(self, x):
        super(CoercedList, self).append(self._coerce_item(len(self) + 1, x))

    def extend(self, t):
        coerced_items = [self._coerce_item(len(self) + index, item) for index, item in enumerate(t)]
        super(CoercedList, self).extend(coerced_items)

    def insert(self, i, x):
        super(CoercedList, self).insert(i, self._coerce_item(i, x))

    def __iadd__(self, y):
        coerced_items = [self._coerce_item(len(self) + index, item) for index, item in enumerate(y)]
        return super(CoercedList, self).__iadd__(coerced_items)

    def __setslice__(self, i, j, y):
        coerced_items = [self._coerce_item(i + index, item) for index, item in enumerate(y)]
        return super(CoercedList, self).__setslice__(i, j, coerced_items)