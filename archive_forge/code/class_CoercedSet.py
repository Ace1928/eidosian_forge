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
class CoercedSet(CoercedCollectionMixin, set):
    """Set which coerces its values

    Dict implementation which overrides all element-adding methods and
    coercing the element(s) being added to the required element type
    """

    def _coerce_element(self, element):
        if hasattr(self, '_element_type') and self._element_type is not None:
            return self._element_type.coerce(self._obj, '%s[%s]' % (self._field, element), element)
        else:
            return element

    def _coerce_iterable(self, values):
        coerced = set()
        for element in values:
            coerced.add(self._coerce_element(element))
        return coerced

    def add(self, value):
        return super(CoercedSet, self).add(self._coerce_element(value))

    def update(self, values):
        return super(CoercedSet, self).update(self._coerce_iterable(values))

    def symmetric_difference_update(self, values):
        return super(CoercedSet, self).symmetric_difference_update(self._coerce_iterable(values))

    def __ior__(self, y):
        return super(CoercedSet, self).__ior__(self._coerce_iterable(y))

    def __ixor__(self, y):
        return super(CoercedSet, self).__ixor__(self._coerce_iterable(y))