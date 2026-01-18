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
class SensitiveString(String):
    """A string field type that may contain sensitive (password) information.

    Passwords in the string value are masked when stringified.
    """

    def stringify(self, value):
        return super(SensitiveString, self).stringify(strutils.mask_password(value))