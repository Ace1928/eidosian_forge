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
class StringPattern(FieldType):

    def get_schema(self):
        if hasattr(self, 'PATTERN'):
            return {'type': ['string'], 'pattern': self.PATTERN}
        else:
            msg = _('%s has no pattern') % self.__class__.__name__
            raise AttributeError(msg)