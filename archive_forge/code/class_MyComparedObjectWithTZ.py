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
@base.VersionedObjectRegistry.register_if(False)
class MyComparedObjectWithTZ(base.VersionedObject):
    fields = {'tzfield': fields.DateTimeField()}