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
def _canonicalize_args(self, context, args, kwargs):
    args = tuple([self._ser.deserialize_entity(context, self._ser.serialize_entity(context, arg)) for arg in args])
    kwargs = dict([(argname, self._ser.deserialize_entity(context, self._ser.serialize_entity(context, arg))) for argname, arg in kwargs.items()])
    return (args, kwargs)