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
class BaseEnumField(AutoTypedField):
    """Base class for all enum field types

    This class should not be directly instantiated. Instead
    subclass it and set AUTO_TYPE to be a SomeEnum()
    where SomeEnum is a subclass of Enum.
    """

    def __init__(self, **kwargs):
        if self.AUTO_TYPE is None:
            raise exception.EnumFieldUnset(fieldname=self.__class__.__name__)
        if not isinstance(self.AUTO_TYPE, Enum):
            raise exception.EnumFieldInvalid(typename=self.AUTO_TYPE.__class__.__name__, fieldname=self.__class__.__name__)
        super(BaseEnumField, self).__init__(**kwargs)

    def __repr__(self):
        valid_values = self._type.valid_values
        args = {'nullable': self._nullable, 'default': self._default}
        args.update({'valid_values': valid_values})
        return '%s(%s)' % (self._type.__class__.__name__, ','.join(['%s=%s' % (k, v) for k, v in sorted(args.items())]))

    @property
    def valid_values(self):
        """Return the list of valid values for the field."""
        return self._type.valid_values