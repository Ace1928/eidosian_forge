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
class TestObjectSerializer(_BaseTestCase):

    def test_serialize_entity_primitive(self):
        ser = base.VersionedObjectSerializer()
        for thing in (1, 'foo', [1, 2], {'foo': 'bar'}):
            self.assertEqual(thing, ser.serialize_entity(None, thing))

    def test_deserialize_entity_primitive(self):
        ser = base.VersionedObjectSerializer()
        for thing in (1, 'foo', [1, 2], {'foo': 'bar'}):
            self.assertEqual(thing, ser.deserialize_entity(None, thing))

    def test_serialize_set_to_list(self):
        ser = base.VersionedObjectSerializer()
        self.assertEqual([1, 2], ser.serialize_entity(None, set([1, 2])))

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

    def test_deserialize_entity_newer_version_backports(self):
        self._test_deserialize_entity_newer('1.25', '1.6')

    def test_deserialize_entity_newer_revision_does_not_backport_zero(self):
        self._test_deserialize_entity_newer('1.6.0', None)

    def test_deserialize_entity_newer_revision_does_not_backport(self):
        self._test_deserialize_entity_newer('1.6.1', None)

    def test_deserialize_entity_newer_version_passes_revision(self):
        self._test_deserialize_entity_newer('1.7', '1.6.1', my_version='1.6.1')

    def test_deserialize_dot_z_with_extra_stuff(self):
        primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.6.1', 'versioned_object.data': {'foo': 1, 'unexpected_thing': 'foobar'}}
        ser = base.VersionedObjectSerializer()
        obj = ser.deserialize_entity(self.context, primitive)
        self.assertEqual(1, obj.foo)
        self.assertFalse(hasattr(obj, 'unexpected_thing'))
        self.assertEqual('1.6', obj.VERSION)

    def test_deserialize_entity_newer_version_no_indirection(self):
        ser = base.VersionedObjectSerializer()
        obj = MyObj()
        obj.VERSION = '1.25'
        primitive = obj.obj_to_primitive()
        self.assertRaises(exception.IncompatibleObjectVersion, ser.deserialize_entity, self.context, primitive)

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

    def test_nested_backport_new_method(self):
        self._test_nested_backport(old=False)

    def test_nested_backport_old_method(self):
        self._test_nested_backport(old=True)

    def test_object_serialization(self):
        ser = base.VersionedObjectSerializer()
        obj = MyObj()
        primitive = ser.serialize_entity(self.context, obj)
        self.assertIn('versioned_object.name', primitive)
        obj2 = ser.deserialize_entity(self.context, primitive)
        self.assertIsInstance(obj2, MyObj)
        self.assertEqual(self.context, obj2._context)

    def test_object_serialization_iterables(self):
        ser = base.VersionedObjectSerializer()
        obj = MyObj()
        for iterable in (list, tuple, set):
            thing = iterable([obj])
            primitive = ser.serialize_entity(self.context, thing)
            self.assertEqual(1, len(primitive))
            for item in primitive:
                self.assertNotIsInstance(item, base.VersionedObject)
            thing2 = ser.deserialize_entity(self.context, primitive)
            self.assertEqual(1, len(thing2))
            for item in thing2:
                self.assertIsInstance(item, MyObj)
        thing = {'key': obj}
        primitive = ser.serialize_entity(self.context, thing)
        self.assertEqual(1, len(primitive))
        for item in primitive.values():
            self.assertNotIsInstance(item, base.VersionedObject)
        thing2 = ser.deserialize_entity(self.context, primitive)
        self.assertEqual(1, len(thing2))
        for item in thing2.values():
            self.assertIsInstance(item, MyObj)
        thing = {'foo': obj.obj_to_primitive()}
        primitive = ser.serialize_entity(self.context, thing)
        self.assertEqual(thing, primitive)
        thing2 = ser.deserialize_entity(self.context, thing)
        self.assertIsInstance(thing2['foo'], base.VersionedObject)

    def test_serializer_subclass_namespace(self):

        @base.VersionedObjectRegistry.register
        class MyNSObj(base.VersionedObject):
            OBJ_SERIAL_NAMESPACE = 'foo'
            fields = {'foo': fields.IntegerField()}

        class MySerializer(base.VersionedObjectSerializer):
            OBJ_BASE_CLASS = MyNSObj
        ser = MySerializer()
        obj = MyNSObj(foo=123)
        obj2 = ser.deserialize_entity(None, ser.serialize_entity(None, obj))
        self.assertIsInstance(obj2, MyNSObj)
        self.assertEqual(obj.foo, obj2.foo)

    def test_serializer_subclass_namespace_mismatch(self):

        @base.VersionedObjectRegistry.register
        class MyNSObj(base.VersionedObject):
            OBJ_SERIAL_NAMESPACE = 'foo'
            fields = {'foo': fields.IntegerField()}

        class MySerializer(base.VersionedObjectSerializer):
            OBJ_BASE_CLASS = MyNSObj
        myser = MySerializer()
        voser = base.VersionedObjectSerializer()
        obj = MyObj(foo=123)
        obj2 = myser.deserialize_entity(None, voser.serialize_entity(None, obj))
        self.assertNotIsInstance(obj2, MyNSObj)
        self.assertIn('versioned_object.name', obj2)

    def test_serializer_subclass_base_object_indirection(self):

        @base.VersionedObjectRegistry.register
        class MyNSObj(base.VersionedObject):
            OBJ_SERIAL_NAMESPACE = 'foo'
            fields = {'foo': fields.IntegerField()}
            indirection_api = mock.MagicMock()

        class MySerializer(base.VersionedObjectSerializer):
            OBJ_BASE_CLASS = MyNSObj
        ser = MySerializer()
        prim = MyNSObj(foo=1).obj_to_primitive()
        prim['foo.version'] = '2.0'
        ser.deserialize_entity(mock.sentinel.context, prim)
        indirection_api = MyNSObj.indirection_api
        indirection_api.object_backport_versions.assert_called_once_with(mock.sentinel.context, prim, {'MyNSObj': '1.0'})

    @mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api')
    def test_serializer_calls_old_backport_interface(self, indirection_api):

        @base.VersionedObjectRegistry.register
        class MyOldObj(base.VersionedObject):
            pass
        ser = base.VersionedObjectSerializer()
        prim = MyOldObj(foo=1).obj_to_primitive()
        prim['versioned_object.version'] = '2.0'
        indirection_api.object_backport_versions.side_effect = NotImplementedError('Old')
        ser.deserialize_entity(mock.sentinel.context, prim)
        indirection_api.object_backport.assert_called_once_with(mock.sentinel.context, prim, '1.0')