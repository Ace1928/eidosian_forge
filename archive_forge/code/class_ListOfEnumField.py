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
class ListOfEnumField(AutoTypedField):

    def __init__(self, valid_values, **kwargs):
        self.AUTO_TYPE = List(Enum(valid_values))
        super(ListOfEnumField, self).__init__(**kwargs)

    def __repr__(self):
        valid_values = self._type._element_type._type.valid_values
        args = {'nullable': self._nullable, 'default': self._default}
        args.update({'valid_values': valid_values})
        return '%s(%s)' % (self._type.__class__.__name__, ','.join(['%s=%s' % (k, v) for k, v in sorted(args.items())]))