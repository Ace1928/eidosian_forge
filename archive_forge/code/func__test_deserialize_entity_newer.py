import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
@mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api')
def _test_deserialize_entity_newer(self, obj_version, backported_to, mock_iapi, my_version='1.6'):
    ser = base.VersionedObjectSerializer()
    mock_iapi.object_backport_versions.return_value = 'backported'

    @base.VersionedObjectRegistry.register
    class MyTestObj(MyObj):
        VERSION = my_version
    obj = MyTestObj()
    obj.VERSION = obj_version
    primitive = obj.obj_to_primitive()
    result = ser.deserialize_entity(self.context, primitive)
    if backported_to is None:
        self.assertFalse(mock_iapi.object_backport_versions.called)
    else:
        self.assertEqual('backported', result)
        mock_iapi.object_backport_versions.assert_called_with(self.context, primitive, {'MyTestObj': my_version, 'MyOwnedObject': '1.0'})