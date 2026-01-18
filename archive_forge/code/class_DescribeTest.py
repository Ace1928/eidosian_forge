import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class DescribeTest(test_util.TestCase):

    def testModule(self):
        self.assertEquals(descriptor.describe_file(test_util), descriptor.describe(test_util))

    def testField(self):
        self.assertEquals(descriptor.describe_field(test_util.NestedMessage.a_value), descriptor.describe(test_util.NestedMessage.a_value))

    def testEnumValue(self):
        self.assertEquals(descriptor.describe_enum_value(test_util.OptionalMessage.SimpleEnum.VAL1), descriptor.describe(test_util.OptionalMessage.SimpleEnum.VAL1))

    def testMessage(self):
        self.assertEquals(descriptor.describe_message(test_util.NestedMessage), descriptor.describe(test_util.NestedMessage))

    def testEnum(self):
        self.assertEquals(descriptor.describe_enum(test_util.OptionalMessage.SimpleEnum), descriptor.describe(test_util.OptionalMessage.SimpleEnum))

    def testUndescribable(self):

        class NonService(object):

            def fn(self):
                pass
        for value in (NonService, NonService.fn, 1, 'string', 1.2, None):
            self.assertEquals(None, descriptor.describe(value))