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
def _test_nested_backport(self, old):

    @base.VersionedObjectRegistry.register
    class Parent(base.VersionedObject):
        VERSION = '1.0'
        fields = {'child': fields.ObjectField('MyObj')}

    @base.VersionedObjectRegistry.register
    class Parent(base.VersionedObject):
        VERSION = '1.1'
        fields = {'child': fields.ObjectField('MyObj')}
    child = MyObj(foo=1)
    parent = Parent(child=child)
    prim = parent.obj_to_primitive()
    child_prim = prim['versioned_object.data']['child']
    child_prim['versioned_object.version'] = '1.10'
    ser = base.VersionedObjectSerializer()
    with mock.patch.object(base.VersionedObject, 'indirection_api') as a:
        if old:
            a.object_backport_versions.side_effect = NotImplementedError
        ser.deserialize_entity(self.context, prim)
        a.object_backport_versions.assert_called_once_with(self.context, prim, {'Parent': '1.1', 'MyObj': '1.6', 'MyOwnedObject': '1.0'})
        if old:
            a.object_backport.assert_called_once_with(self.context, prim, '1.1')