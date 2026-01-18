from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
def _doit(obj, *args, **kwargs):
    result = self._original_otp(obj, *args, **kwargs)
    changes_key = obj._obj_primitive_key('changes')
    if changes_key in result:
        result[changes_key].sort()
    return result