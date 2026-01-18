import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
class ClassWithDefaultedSetField(base.VersionedObject):
    VERSION = 1.0
    fields = {'empty_default': fields.SetOfIntegersField(default=set()), 'non_empty_default': fields.SetOfIntegersField(default={1, 2})}