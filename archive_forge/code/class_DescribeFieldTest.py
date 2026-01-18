import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class DescribeFieldTest(test_util.TestCase):

    def testLabel(self):
        for repeated, required, expected_label in ((True, False, descriptor.FieldDescriptor.Label.REPEATED), (False, True, descriptor.FieldDescriptor.Label.REQUIRED), (False, False, descriptor.FieldDescriptor.Label.OPTIONAL)):
            field = messages.IntegerField(10, required=required, repeated=repeated)
            field.name = 'a_field'
            expected = descriptor.FieldDescriptor()
            expected.name = 'a_field'
            expected.number = 10
            expected.label = expected_label
            expected.variant = descriptor.FieldDescriptor.Variant.INT64
            described = descriptor.describe_field(field)
            described.check_initialized()
            self.assertEquals(expected, described)

    def testDefault(self):
        test_cases = ((messages.IntegerField, 200, '200'), (messages.FloatField, 1.5, '1.5'), (messages.FloatField, 1000000.0, '1000000.0'), (messages.BooleanField, True, 'true'), (messages.BooleanField, False, 'false'), (messages.BytesField, b''.join([six.int2byte(x) for x in (31, 32, 33)]), b'\\x1f !'), (messages.StringField, RUSSIA, RUSSIA))
        for field_class, default, expected_default in test_cases:
            field = field_class(10, default=default)
            field.name = u'a_field'
            expected = descriptor.FieldDescriptor()
            expected.name = u'a_field'
            expected.number = 10
            expected.label = descriptor.FieldDescriptor.Label.OPTIONAL
            expected.variant = field_class.DEFAULT_VARIANT
            expected.default_value = expected_default
            described = descriptor.describe_field(field)
            described.check_initialized()
            self.assertEquals(expected, described)

    def testDefault_EnumField(self):

        class MyEnum(messages.Enum):
            VAL = 1
        module_name = test_util.get_module_name(MyEnum)
        field = messages.EnumField(MyEnum, 10, default=MyEnum.VAL)
        field.name = 'a_field'
        expected = descriptor.FieldDescriptor()
        expected.name = 'a_field'
        expected.number = 10
        expected.label = descriptor.FieldDescriptor.Label.OPTIONAL
        expected.variant = messages.EnumField.DEFAULT_VARIANT
        expected.type_name = '%s.MyEnum' % module_name
        expected.default_value = '1'
        described = descriptor.describe_field(field)
        self.assertEquals(expected, described)

    def testMessageField(self):
        field = messages.MessageField(descriptor.FieldDescriptor, 10)
        field.name = 'a_field'
        expected = descriptor.FieldDescriptor()
        expected.name = 'a_field'
        expected.number = 10
        expected.label = descriptor.FieldDescriptor.Label.OPTIONAL
        expected.variant = messages.MessageField.DEFAULT_VARIANT
        expected.type_name = 'apitools.base.protorpclite.descriptor.FieldDescriptor'
        described = descriptor.describe_field(field)
        described.check_initialized()
        self.assertEquals(expected, described)

    def testDateTimeField(self):
        field = message_types.DateTimeField(20)
        field.name = 'a_timestamp'
        expected = descriptor.FieldDescriptor()
        expected.name = 'a_timestamp'
        expected.number = 20
        expected.label = descriptor.FieldDescriptor.Label.OPTIONAL
        expected.variant = messages.MessageField.DEFAULT_VARIANT
        expected.type_name = 'apitools.base.protorpclite.message_types.DateTimeMessage'
        described = descriptor.describe_field(field)
        described.check_initialized()
        self.assertEquals(expected, described)