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
class TestObjMakeList(test.TestCase):

    def test_obj_make_list(self):

        @base.VersionedObjectRegistry.register
        class MyList(base.ObjectListBase, base.VersionedObject):
            fields = {'objects': fields.ListOfObjectsField('MyObj')}
        db_objs = [{'foo': 1, 'bar': 'baz', 'missing': 'banana'}, {'foo': 2, 'bar': 'bat', 'missing': 'apple'}]
        mylist = base.obj_make_list('ctxt', MyList(), MyObj, db_objs)
        self.assertEqual(2, len(mylist))
        self.assertEqual('ctxt', mylist._context)
        for index, item in enumerate(mylist):
            self.assertEqual(db_objs[index]['foo'], item.foo)
            self.assertEqual(db_objs[index]['bar'], item.bar)
            self.assertEqual(db_objs[index]['missing'], item.missing)