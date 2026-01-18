import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
class ProtojsonTest(test_util.TestCase, test_util.ProtoConformanceTestBase):
    """Test JSON encoding and decoding."""
    PROTOLIB = protojson

    def CompareEncoded(self, expected_encoded, actual_encoded):
        """JSON encoding will be laundered to remove string differences."""
        self.assertEquals(json.loads(expected_encoded), json.loads(actual_encoded))
    encoded_empty_message = '{}'
    encoded_partial = '{\n    "double_value": 1.23,\n    "int64_value": -100000000000,\n    "int32_value": 1020,\n    "string_value": "a string",\n    "enum_value": "VAL2"\n    }\n    '
    encoded_full = '{\n    "double_value": 1.23,\n    "float_value": -2.5,\n    "int64_value": -100000000000,\n    "uint64_value": 102020202020,\n    "int32_value": 1020,\n    "bool_value": true,\n    "string_value": "a stringя",\n    "bytes_value": "YSBieXRlc//+",\n    "enum_value": "VAL2"\n    }\n    '
    encoded_repeated = '{\n    "double_value": [1.23, 2.3],\n    "float_value": [-2.5, 0.5],\n    "int64_value": [-100000000000, 20],\n    "uint64_value": [102020202020, 10],\n    "int32_value": [1020, 718],\n    "bool_value": [true, false],\n    "string_value": ["a stringя", "another string"],\n    "bytes_value": ["YSBieXRlc//+", "YW5vdGhlciBieXRlcw=="],\n    "enum_value": ["VAL2", "VAL1"]\n    }\n    '
    encoded_nested = '{\n    "nested": {\n      "a_value": "a string"\n    }\n    }\n    '
    encoded_repeated_nested = '{\n    "repeated_nested": [{"a_value": "a string"},\n                        {"a_value": "another string"}]\n    }\n    '
    unexpected_tag_message = '{"unknown": "value"}'
    encoded_default_assigned = '{"a_value": "a default"}'
    encoded_nested_empty = '{"nested": {}}'
    encoded_repeated_nested_empty = '{"repeated_nested": [{}, {}]}'
    encoded_extend_message = '{"int64_value": [400, 50, 6000]}'
    encoded_string_types = '{"string_value": "Latin"}'
    encoded_invalid_enum = '{"enum_value": "undefined"}'

    def testConvertIntegerToFloat(self):
        """Test that integers passed in to float fields are converted.

        This is necessary because JSON outputs integers for numbers
        with 0 decimals.

        """
        message = protojson.decode_message(MyMessage, '{"a_float": 10}')
        self.assertTrue(isinstance(message.a_float, float))
        self.assertEquals(10.0, message.a_float)

    def testConvertStringToNumbers(self):
        """Test that strings passed to integer fields are converted."""
        message = protojson.decode_message(MyMessage, '{"an_integer": "10",\n                                           "a_float": "3.5",\n                                           "a_repeated": ["1", "2"],\n                                           "a_repeated_float": ["1.5", "2", 10]\n                                           }')
        self.assertEquals(MyMessage(an_integer=10, a_float=3.5, a_repeated=[1, 2], a_repeated_float=[1.5, 2.0, 10.0]), message)

    def testWrongTypeAssignment(self):
        """Test when wrong type is assigned to a field."""
        self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"a_string": 10}')
        self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"an_integer": 10.2}')
        self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"an_integer": "10.2"}')

    def testNumericEnumeration(self):
        """Test that numbers work for enum values."""
        message = protojson.decode_message(MyMessage, '{"an_enum": 2}')
        expected_message = MyMessage()
        expected_message.an_enum = MyMessage.Color.GREEN
        self.assertEquals(expected_message, message)

    def testNumericEnumerationNegativeTest(self):
        """Test with an invalid number for the enum value."""
        message = protojson.decode_message(MyMessage, '{"an_enum": 89}')
        expected_message = MyMessage()
        self.assertEquals(expected_message, message)
        self.assertEquals('{"an_enum": 89}', protojson.encode_message(message))

    def testAlphaEnumeration(self):
        """Test that alpha enum values work."""
        message = protojson.decode_message(MyMessage, '{"an_enum": "RED"}')
        expected_message = MyMessage()
        expected_message.an_enum = MyMessage.Color.RED
        self.assertEquals(expected_message, message)

    def testAlphaEnumerationNegativeTest(self):
        """The alpha enum value is invalid."""
        message = protojson.decode_message(MyMessage, '{"an_enum": "IAMINVALID"}')
        expected_message = MyMessage()
        self.assertEquals(expected_message, message)
        self.assertEquals('{"an_enum": "IAMINVALID"}', protojson.encode_message(message))

    def testEnumerationNegativeTestWithEmptyString(self):
        """The enum value is an empty string."""
        message = protojson.decode_message(MyMessage, '{"an_enum": ""}')
        expected_message = MyMessage()
        self.assertEquals(expected_message, message)
        self.assertEquals('{"an_enum": ""}', protojson.encode_message(message))

    def testNullValues(self):
        """Test that null values overwrite existing values."""
        self.assertEquals(MyMessage(), protojson.decode_message(MyMessage, '{"an_integer": null, "a_nested": null, "an_enum": null}'))

    def testEmptyList(self):
        """Test that empty lists are ignored."""
        self.assertEquals(MyMessage(), protojson.decode_message(MyMessage, '{"a_repeated": []}'))

    def testNotJSON(self):
        """Test error when string is not valid JSON."""
        self.assertRaises(ValueError, protojson.decode_message, MyMessage, '{this is not json}')

    def testDoNotEncodeStrangeObjects(self):
        """Test trying to encode a strange object.

        The main purpose of this test is to complete coverage. It
        ensures that the default behavior of the JSON encoder is
        preserved when someone tries to serialized an unexpected type.

        """

        class BogusObject(object):

            def check_initialized(self):
                pass
        self.assertRaises(TypeError, protojson.encode_message, BogusObject())

    def testMergeEmptyString(self):
        """Test merging the empty or space only string."""
        message = protojson.decode_message(test_util.OptionalMessage, '')
        self.assertEquals(test_util.OptionalMessage(), message)
        message = protojson.decode_message(test_util.OptionalMessage, ' ')
        self.assertEquals(test_util.OptionalMessage(), message)

    def testProtojsonUnrecognizedFieldName(self):
        """Test that unrecognized fields are saved and can be accessed."""
        decoded = protojson.decode_message(MyMessage, '{"an_integer": 1, "unknown_val": 2}')
        self.assertEquals(decoded.an_integer, 1)
        self.assertEquals(1, len(decoded.all_unrecognized_fields()))
        self.assertEquals('unknown_val', decoded.all_unrecognized_fields()[0])
        self.assertEquals((2, messages.Variant.INT64), decoded.get_unrecognized_field_info('unknown_val'))

    def testProtojsonUnrecognizedFieldNumber(self):
        """Test that unrecognized fields are saved and can be accessed."""
        decoded = protojson.decode_message(MyMessage, '{"an_integer": 1, "1001": "unknown", "-123": "negative", "456_mixed": 2}')
        self.assertEquals(decoded.an_integer, 1)
        self.assertEquals(3, len(decoded.all_unrecognized_fields()))
        self.assertFalse(1001 in decoded.all_unrecognized_fields())
        self.assertTrue('1001' in decoded.all_unrecognized_fields())
        self.assertEquals(('unknown', messages.Variant.STRING), decoded.get_unrecognized_field_info('1001'))
        self.assertTrue('-123' in decoded.all_unrecognized_fields())
        self.assertEquals(('negative', messages.Variant.STRING), decoded.get_unrecognized_field_info('-123'))
        self.assertTrue('456_mixed' in decoded.all_unrecognized_fields())
        self.assertEquals((2, messages.Variant.INT64), decoded.get_unrecognized_field_info('456_mixed'))

    def testProtojsonUnrecognizedNull(self):
        """Test that unrecognized fields that are None are skipped."""
        decoded = protojson.decode_message(MyMessage, '{"an_integer": 1, "unrecognized_null": null}')
        self.assertEquals(decoded.an_integer, 1)
        self.assertEquals(decoded.all_unrecognized_fields(), [])

    def testUnrecognizedFieldVariants(self):
        """Test that unrecognized fields are mapped to the right variants."""
        for encoded, expected_variant in (('{"an_integer": 1, "unknown_val": 2}', messages.Variant.INT64), ('{"an_integer": 1, "unknown_val": 2.0}', messages.Variant.DOUBLE), ('{"an_integer": 1, "unknown_val": "string value"}', messages.Variant.STRING), ('{"an_integer": 1, "unknown_val": [1, 2, 3]}', messages.Variant.INT64), ('{"an_integer": 1, "unknown_val": [1, 2.0, 3]}', messages.Variant.DOUBLE), ('{"an_integer": 1, "unknown_val": [1, "foo", 3]}', messages.Variant.STRING), ('{"an_integer": 1, "unknown_val": true}', messages.Variant.BOOL)):
            decoded = protojson.decode_message(MyMessage, encoded)
            self.assertEquals(decoded.an_integer, 1)
            self.assertEquals(1, len(decoded.all_unrecognized_fields()))
            self.assertEquals('unknown_val', decoded.all_unrecognized_fields()[0])
            _, decoded_variant = decoded.get_unrecognized_field_info('unknown_val')
            self.assertEquals(expected_variant, decoded_variant)

    def testDecodeDateTime(self):
        for datetime_string, datetime_vals in (('2012-09-30T15:31:50.262', (2012, 9, 30, 15, 31, 50, 262000)), ('2012-09-30T15:31:50', (2012, 9, 30, 15, 31, 50, 0))):
            message = protojson.decode_message(MyMessage, '{"a_datetime": "%s"}' % datetime_string)
            expected_message = MyMessage(a_datetime=datetime.datetime(*datetime_vals))
            self.assertEquals(expected_message, message)

    def testDecodeInvalidDateTime(self):
        self.assertRaises(messages.DecodeError, protojson.decode_message, MyMessage, '{"a_datetime": "invalid"}')

    def testDecodeInvalidMessage(self):
        encoded = '{\n        "a_nested_datetime": {\n          "nested_dt_value": "invalid"\n          }\n        }\n        '
        self.assertRaises(messages.DecodeError, protojson.decode_message, MyMessage, encoded)

    def testEncodeDateTime(self):
        for datetime_string, datetime_vals in (('2012-09-30T15:31:50.262000', (2012, 9, 30, 15, 31, 50, 262000)), ('2012-09-30T15:31:50.262123', (2012, 9, 30, 15, 31, 50, 262123)), ('2012-09-30T15:31:50', (2012, 9, 30, 15, 31, 50, 0))):
            decoded_message = protojson.encode_message(MyMessage(a_datetime=datetime.datetime(*datetime_vals)))
            expected_decoding = '{"a_datetime": "%s"}' % datetime_string
            self.CompareEncoded(expected_decoding, decoded_message)

    def testDecodeRepeatedDateTime(self):
        message = protojson.decode_message(MyMessage, '{"a_repeated_datetime": ["2012-09-30T15:31:50.262", "2010-01-21T09:52:00", "2000-01-01T01:00:59.999999"]}')
        expected_message = MyMessage(a_repeated_datetime=[datetime.datetime(2012, 9, 30, 15, 31, 50, 262000), datetime.datetime(2010, 1, 21, 9, 52), datetime.datetime(2000, 1, 1, 1, 0, 59, 999999)])
        self.assertEquals(expected_message, message)

    def testDecodeCustom(self):
        message = protojson.decode_message(MyMessage, '{"a_custom": 1}')
        self.assertEquals(MyMessage(a_custom=1), message)

    def testDecodeInvalidCustom(self):
        self.assertRaises(messages.ValidationError, protojson.decode_message, MyMessage, '{"a_custom": "invalid"}')

    def testEncodeCustom(self):
        decoded_message = protojson.encode_message(MyMessage(a_custom=1))
        self.CompareEncoded('{"a_custom": 1}', decoded_message)

    def testDecodeRepeatedCustom(self):
        message = protojson.decode_message(MyMessage, '{"a_repeated_custom": [1, 2, 3]}')
        self.assertEquals(MyMessage(a_repeated_custom=[1, 2, 3]), message)

    def testDecodeRepeatedEmpty(self):
        message = protojson.decode_message(MyMessage, '{"a_repeated": []}')
        self.assertEquals(MyMessage(a_repeated=[]), message)

    def testDecodeNone(self):
        message = protojson.decode_message(MyMessage, '{"an_integer": []}')
        self.assertEquals(MyMessage(an_integer=None), message)

    def testDecodeBadBase64BytesField(self):
        """Test decoding improperly encoded base64 bytes value."""
        self.assertRaisesWithRegexpMatch(messages.DecodeError, 'Base64 decoding error: Incorrect padding', protojson.decode_message, test_util.OptionalMessage, '{"bytes_value": "abcdefghijklmnopq"}')