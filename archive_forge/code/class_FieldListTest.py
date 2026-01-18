import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class FieldListTest(test_util.TestCase):

    def setUp(self):
        self.integer_field = messages.IntegerField(1, repeated=True)

    def testConstructor(self):
        self.assertEquals([1, 2, 3], messages.FieldList(self.integer_field, [1, 2, 3]))
        self.assertEquals([1, 2, 3], messages.FieldList(self.integer_field, (1, 2, 3)))
        self.assertEquals([], messages.FieldList(self.integer_field, []))

    def testNone(self):
        self.assertRaises(TypeError, messages.FieldList, self.integer_field, None)

    def testDoNotAutoConvertString(self):
        string_field = messages.StringField(1, repeated=True)
        self.assertRaises(messages.ValidationError, messages.FieldList, string_field, 'abc')

    def testConstructorCopies(self):
        a_list = [1, 3, 6]
        field_list = messages.FieldList(self.integer_field, a_list)
        self.assertFalse(a_list is field_list)
        self.assertFalse(field_list is messages.FieldList(self.integer_field, field_list))

    def testNonRepeatedField(self):
        self.assertRaisesWithRegexpMatch(messages.FieldDefinitionError, 'FieldList may only accept repeated fields', messages.FieldList, messages.IntegerField(1), [])

    def testConstructor_InvalidValues(self):
        self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 1 (type %r)' % (six.integer_types, str)), messages.FieldList, self.integer_field, ['1', '2', '3'])

    def testConstructor_Scalars(self):
        self.assertRaisesWithRegexpMatch(messages.ValidationError, 'IntegerField is repeated. Found: 3', messages.FieldList, self.integer_field, 3)
        self.assertRaisesWithRegexpMatch(messages.ValidationError, 'IntegerField is repeated. Found: <(list[_]?|sequence)iterator object', messages.FieldList, self.integer_field, iter([1, 2, 3]))

    def testSetSlice(self):
        field_list = messages.FieldList(self.integer_field, [1, 2, 3, 4, 5])
        field_list[1:3] = [10, 20]
        self.assertEquals([1, 10, 20, 4, 5], field_list)

    def testSetSlice_InvalidValues(self):
        field_list = messages.FieldList(self.integer_field, [1, 2, 3, 4, 5])

        def setslice():
            field_list[1:3] = ['10', '20']
        msg_re = re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str))
        self.assertRaisesWithRegexpMatch(messages.ValidationError, msg_re, setslice)

    def testSetItem(self):
        field_list = messages.FieldList(self.integer_field, [2])
        field_list[0] = 10
        self.assertEquals([10], field_list)

    def testSetItem_InvalidValues(self):
        field_list = messages.FieldList(self.integer_field, [2])

        def setitem():
            field_list[0] = '10'
        self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str)), setitem)

    def testAppend(self):
        field_list = messages.FieldList(self.integer_field, [2])
        field_list.append(10)
        self.assertEquals([2, 10], field_list)

    def testAppend_InvalidValues(self):
        field_list = messages.FieldList(self.integer_field, [2])
        field_list.name = 'a_field'

        def append():
            field_list.append('10')
        self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str)), append)

    def testExtend(self):
        field_list = messages.FieldList(self.integer_field, [2])
        field_list.extend([10])
        self.assertEquals([2, 10], field_list)

    def testExtend_InvalidValues(self):
        field_list = messages.FieldList(self.integer_field, [2])

        def extend():
            field_list.extend(['10'])
        self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str)), extend)

    def testInsert(self):
        field_list = messages.FieldList(self.integer_field, [2, 3])
        field_list.insert(1, 10)
        self.assertEquals([2, 10, 3], field_list)

    def testInsert_InvalidValues(self):
        field_list = messages.FieldList(self.integer_field, [2, 3])

        def insert():
            field_list.insert(1, '10')
        self.assertRaisesWithRegexpMatch(messages.ValidationError, re.escape('Expected type %r for IntegerField, found 10 (type %r)' % (six.integer_types, str)), insert)

    def testPickle(self):
        """Testing pickling and unpickling of FieldList instances."""
        field_list = messages.FieldList(self.integer_field, [1, 2, 3, 4, 5])
        unpickled = pickle.loads(pickle.dumps(field_list))
        self.assertEquals(field_list, unpickled)
        self.assertIsInstance(unpickled.field, messages.IntegerField)
        self.assertEquals(1, unpickled.field.number)
        self.assertTrue(unpickled.field.repeated)