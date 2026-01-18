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
class CoercedCollectionMixin(object):

    def __init__(self, *args, **kwargs):
        self._element_type = None
        self._obj = None
        self._field = None
        super(CoercedCollectionMixin, self).__init__(*args, **kwargs)

    def enable_coercing(self, element_type, obj, field):
        self._element_type = element_type
        self._obj = obj
        self._field = field