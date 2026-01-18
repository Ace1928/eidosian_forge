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
class TestListObjectConcat(test.TestCase):

    def test_list_object_concat(self):

        @base.VersionedObjectRegistry.register_if(False)
        class MyList(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}
        values = [1, 2, 42]
        list1 = MyList(objects=[MyOwnedObject(baz=values[0]), MyOwnedObject(baz=values[1])])
        list2 = MyList(objects=[MyOwnedObject(baz=values[2])])
        concat_list = list1 + list2
        for idx, obj in enumerate(concat_list):
            self.assertEqual(values[idx], obj.baz)
        self.assertEqual(2, len(list1.objects))
        self.assertEqual(1, list1.objects[0].baz)
        self.assertEqual(2, list1.objects[1].baz)
        self.assertEqual(1, len(list2.objects))
        self.assertEqual(42, list2.objects[0].baz)

    def test_list_object_concat_fails_different_objects(self):

        @base.VersionedObjectRegistry.register_if(False)
        class MyList(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}

        @base.VersionedObjectRegistry.register_if(False)
        class MyList2(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}
        list1 = MyList(objects=[MyOwnedObject(baz=1)])
        list2 = MyList2(objects=[MyOwnedObject(baz=2)])

        def add(x, y):
            return x + y
        self.assertRaises(TypeError, add, list1, list2)
        self.assertEqual(1, len(list1.objects))
        self.assertEqual(1, len(list2.objects))
        self.assertEqual(1, list1.objects[0].baz)
        self.assertEqual(2, list2.objects[0].baz)

    def test_list_object_concat_fails_extra_fields(self):

        @base.VersionedObjectRegistry.register_if(False)
        class MyList(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyOwnedObject'), 'foo': fields.IntegerField(nullable=True)}
        list1 = MyList(objects=[MyOwnedObject(baz=1)])
        list2 = MyList(objects=[MyOwnedObject(baz=2)])

        def add(x, y):
            return x + y
        self.assertRaises(TypeError, add, list1, list2)
        self.assertEqual(1, len(list1.objects))
        self.assertEqual(1, len(list2.objects))
        self.assertEqual(1, list1.objects[0].baz)
        self.assertEqual(2, list2.objects[0].baz)

    def test_builtin_list_add_fails(self):

        @base.VersionedObjectRegistry.register_if(False)
        class MyList(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}
        list1 = MyList(objects=[MyOwnedObject(baz=1)])

        def add(obj):
            return obj + []
        self.assertRaises(TypeError, add, list1)

    def test_builtin_list_radd_fails(self):

        @base.VersionedObjectRegistry.register_if(False)
        class MyList(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}
        list1 = MyList(objects=[MyOwnedObject(baz=1)])

        def add(obj):
            return [] + obj
        self.assertRaises(TypeError, add, list1)