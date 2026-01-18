import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
@obj_base.VersionedObjectRegistry.register
class TestFakeObject(obj_base.VersionedObject):
    OBJ_PROJECT_NAMESPACE = 'fake-project'