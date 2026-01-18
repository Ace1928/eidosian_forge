import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class DescribeEnumValueTest(test_util.TestCase):

    def testDescribe(self):

        class MyEnum(messages.Enum):
            MY_NAME = 10
        expected = descriptor.EnumValueDescriptor()
        expected.name = 'MY_NAME'
        expected.number = 10
        described = descriptor.describe_enum_value(MyEnum.MY_NAME)
        described.check_initialized()
        self.assertEquals(expected, described)