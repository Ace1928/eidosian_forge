import datetime
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
class DateTimeFieldTest(test_util.TestCase):

    def testValueToMessage(self):
        field = message_types.DateTimeField(1)
        message = field.value_to_message(datetime.datetime(2033, 2, 4, 11, 22, 10))
        self.assertEqual(message_types.DateTimeMessage(milliseconds=1991128930000), message)

    def testValueToMessageBadValue(self):
        field = message_types.DateTimeField(1)
        self.assertRaisesWithRegexpMatch(messages.EncodeError, 'Expected type datetime, got int: 20', field.value_to_message, 20)

    def testValueToMessageWithTimeZone(self):
        time_zone = util.TimeZoneOffset(60 * 10)
        field = message_types.DateTimeField(1)
        message = field.value_to_message(datetime.datetime(2033, 2, 4, 11, 22, 10, tzinfo=time_zone))
        self.assertEqual(message_types.DateTimeMessage(milliseconds=1991128930000, time_zone_offset=600), message)

    def testValueFromMessage(self):
        message = message_types.DateTimeMessage(milliseconds=1991128000000)
        field = message_types.DateTimeField(1)
        timestamp = field.value_from_message(message)
        self.assertEqual(datetime.datetime(2033, 2, 4, 11, 6, 40), timestamp)

    def testValueFromMessageBadValue(self):
        field = message_types.DateTimeField(1)
        self.assertRaisesWithRegexpMatch(messages.DecodeError, 'Expected type DateTimeMessage, got VoidMessage: <VoidMessage>', field.value_from_message, message_types.VoidMessage())

    def testValueFromMessageWithTimeZone(self):
        message = message_types.DateTimeMessage(milliseconds=1991128000000, time_zone_offset=300)
        field = message_types.DateTimeField(1)
        timestamp = field.value_from_message(message)
        time_zone = util.TimeZoneOffset(60 * 5)
        self.assertEqual(datetime.datetime(2033, 2, 4, 11, 6, 40, tzinfo=time_zone), timestamp)