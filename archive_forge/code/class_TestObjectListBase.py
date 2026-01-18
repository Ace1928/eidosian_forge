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
class TestObjectListBase(test.TestCase):

    def test_list_like_operations(self):

        @base.VersionedObjectRegistry.register
        class MyElement(base.VersionedObject):
            fields = {'foo': fields.IntegerField()}

            def __init__(self, foo):
                super(MyElement, self).__init__()
                self.foo = foo

        class Foo(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyElement')}
        objlist = Foo(context='foo', objects=[MyElement(1), MyElement(2), MyElement(3)])
        self.assertEqual(list(objlist), objlist.objects)
        self.assertEqual(len(objlist), 3)
        self.assertIn(objlist.objects[0], objlist)
        self.assertEqual(list(objlist[:1]), [objlist.objects[0]])
        self.assertEqual(objlist[:1]._context, 'foo')
        self.assertEqual(objlist[2], objlist.objects[2])
        self.assertEqual(objlist.count(objlist.objects[0]), 1)
        self.assertEqual(objlist.index(objlist.objects[1]), 1)
        objlist.sort(key=lambda x: x.foo, reverse=True)
        self.assertEqual([3, 2, 1], [x.foo for x in objlist])

    def test_serialization(self):

        @base.VersionedObjectRegistry.register
        class Foo(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('Bar')}

        @base.VersionedObjectRegistry.register
        class Bar(base.VersionedObject):
            fields = {'foo': fields.Field(fields.String())}
        obj = Foo(objects=[])
        for i in 'abc':
            bar = Bar(foo=i)
            obj.objects.append(bar)
        obj2 = base.VersionedObject.obj_from_primitive(obj.obj_to_primitive())
        self.assertFalse(obj is obj2)
        self.assertEqual([x.foo for x in obj], [y.foo for y in obj2])

    def _test_object_list_version_mappings(self, list_obj_class):
        list_field = list_obj_class.fields['objects']
        item_obj_field = list_field._type._element_type
        item_obj_name = item_obj_field._type._obj_name
        obj_classes = base.VersionedObjectRegistry.obj_classes()[item_obj_name]
        for item_class in obj_classes:
            if is_test_object(item_class):
                continue
            self.assertIn(item_class.VERSION, list_obj_class.child_versions.values(), 'Version mapping is incomplete for %s' % list_obj_class.__name__)

    def test_object_version_mappings(self):
        self.skipTest('this needs to be generalized')
        for obj_classes in base.VersionedObjectRegistry.obj_classes().values():
            for obj_class in obj_classes:
                if issubclass(obj_class, base.ObjectListBase):
                    self._test_object_list_version_mappings(obj_class)

    def test_obj_make_compatible_child_versions(self):

        @base.VersionedObjectRegistry.register
        class MyElement(base.VersionedObject):
            fields = {'foo': fields.IntegerField()}

        @base.VersionedObjectRegistry.register
        class Foo(base.ObjectListBase, base.VersionedObject):
            VERSION = '1.1'
            fields = {'objects': fields.ListOfObjectsField('MyElement')}
            child_versions = {'1.0': '1.0', '1.1': '1.0'}
        subobj = MyElement(foo=1)
        obj = Foo(objects=[subobj])
        primitive = obj.obj_to_primitive()['versioned_object.data']
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            obj.obj_make_compatible(copy.copy(primitive), '1.1')
            self.assertTrue(mock_compat.called)

    def test_obj_make_compatible_obj_relationships(self):

        @base.VersionedObjectRegistry.register
        class MyElement(base.VersionedObject):
            fields = {'foo': fields.IntegerField()}

        @base.VersionedObjectRegistry.register
        class Bar(base.ObjectListBase, base.VersionedObject):
            VERSION = '1.1'
            fields = {'objects': fields.ListOfObjectsField('MyElement')}
            obj_relationships = {'objects': [('1.0', '1.0'), ('1.1', '1.0')]}
        subobj = MyElement(foo=1)
        obj = Bar(objects=[subobj])
        primitive = obj.obj_to_primitive()['versioned_object.data']
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            obj.obj_make_compatible(copy.copy(primitive), '1.1')
            self.assertTrue(mock_compat.called)

    def test_obj_make_compatible_no_relationships(self):

        @base.VersionedObjectRegistry.register
        class MyElement(base.VersionedObject):
            fields = {'foo': fields.IntegerField()}

        @base.VersionedObjectRegistry.register
        class Baz(base.ObjectListBase, base.VersionedObject):
            VERSION = '1.1'
            fields = {'objects': fields.ListOfObjectsField('MyElement')}
        subobj = MyElement(foo=1)
        obj = Baz(objects=[subobj])
        primitive = obj.obj_to_primitive()['versioned_object.data']
        with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
            obj.obj_make_compatible(copy.copy(primitive), '1.1')
            self.assertTrue(mock_compat.called)

    def test_list_changes(self):

        @base.VersionedObjectRegistry.register
        class Foo(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('Bar')}

        @base.VersionedObjectRegistry.register
        class Bar(base.VersionedObject):
            fields = {'foo': fields.StringField()}
        obj = Foo(objects=[])
        self.assertEqual(set(['objects']), obj.obj_what_changed())
        obj.objects.append(Bar(foo='test'))
        self.assertEqual(set(['objects']), obj.obj_what_changed())
        obj.obj_reset_changes()
        self.assertEqual(set(['objects']), obj.obj_what_changed())
        obj.objects[0].obj_reset_changes()
        self.assertEqual(set(), obj.obj_what_changed())

    def test_initialize_objects(self):

        class Foo(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('Bar')}

        class Bar(base.VersionedObject):
            fields = {'foo': fields.StringField()}
        obj = Foo()
        self.assertEqual([], obj.objects)
        self.assertEqual(set(), obj.obj_what_changed())

    def test_obj_repr(self):

        @base.VersionedObjectRegistry.register
        class Foo(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('Bar')}

        @base.VersionedObjectRegistry.register
        class Bar(base.VersionedObject):
            fields = {'uuid': fields.StringField()}
        obj = Foo(objects=[Bar(uuid='fake-uuid')])
        self.assertEqual('Foo(objects=[Bar(fake-uuid)])', repr(obj))