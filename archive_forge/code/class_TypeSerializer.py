from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
class TypeSerializer(object):
    """This class serializes Python data types to DynamoDB types."""

    def serialize(self, value):
        """The method to serialize the Python data types.

        :param value: A python value to be serialized to DynamoDB. Here are
            the various conversions:

            Python                                  DynamoDB
            ------                                  --------
            None                                    {'NULL': True}
            True/False                              {'BOOL': True/False}
            int/Decimal                             {'N': str(value)}
            string                                  {'S': string}
            Binary/bytearray/bytes (py3 only)       {'B': bytes}
            set([int/Decimal])                      {'NS': [str(value)]}
            set([string])                           {'SS': [string])
            set([Binary/bytearray/bytes])           {'BS': [bytes]}
            list                                    {'L': list}
            dict                                    {'M': dict}

            For types that involve numbers, it is recommended that ``Decimal``
            objects are used to be able to round-trip the Python type.
            For types that involve binary, it is recommended that ``Binary``
            objects are used to be able to round-trip the Python type.

        :rtype: dict
        :returns: A dictionary that represents a dynamoDB data type. These
            dictionaries can be directly passed to botocore methods.
        """
        dynamodb_type = self._get_dynamodb_type(value)
        serializer = getattr(self, '_serialize_%s' % dynamodb_type.lower())
        return {dynamodb_type: serializer(value)}

    def _get_dynamodb_type(self, value):
        dynamodb_type = None
        if self._is_null(value):
            dynamodb_type = NULL
        elif self._is_boolean(value):
            dynamodb_type = BOOLEAN
        elif self._is_number(value):
            dynamodb_type = NUMBER
        elif self._is_string(value):
            dynamodb_type = STRING
        elif self._is_binary(value):
            dynamodb_type = BINARY
        elif self._is_type_set(value, self._is_number):
            dynamodb_type = NUMBER_SET
        elif self._is_type_set(value, self._is_string):
            dynamodb_type = STRING_SET
        elif self._is_type_set(value, self._is_binary):
            dynamodb_type = BINARY_SET
        elif self._is_map(value):
            dynamodb_type = MAP
        elif self._is_list(value):
            dynamodb_type = LIST
        else:
            msg = 'Unsupported type "%s" for value "%s"' % (type(value), value)
            raise TypeError(msg)
        return dynamodb_type

    def _is_null(self, value):
        if value is None:
            return True
        return False

    def _is_boolean(self, value):
        if isinstance(value, bool):
            return True
        return False

    def _is_number(self, value):
        if isinstance(value, (six.integer_types, Decimal)):
            return True
        elif isinstance(value, float):
            raise TypeError('Float types are not supported. Use Decimal types instead.')
        return False

    def _is_string(self, value):
        if isinstance(value, six.string_types):
            return True
        return False

    def _is_binary(self, value):
        if isinstance(value, Binary):
            return True
        elif isinstance(value, bytearray):
            return True
        elif six.PY3 and isinstance(value, six.binary_type):
            return True
        return False

    def _is_set(self, value):
        if isinstance(value, collections_abc.Set):
            return True
        return False

    def _is_type_set(self, value, type_validator):
        if self._is_set(value):
            if False not in map(type_validator, value):
                return True
        return False

    def _is_map(self, value):
        if isinstance(value, collections_abc.Mapping):
            return True
        return False

    def _is_list(self, value):
        if isinstance(value, list):
            return True
        return False

    def _serialize_null(self, value):
        return True

    def _serialize_bool(self, value):
        return value

    def _serialize_n(self, value):
        number = str(DYNAMODB_CONTEXT.create_decimal(value))
        if number in ['Infinity', 'NaN']:
            raise TypeError('Infinity and NaN not supported')
        return number

    def _serialize_s(self, value):
        return value

    def _serialize_b(self, value):
        if isinstance(value, Binary):
            value = value.value
        return value

    def _serialize_ss(self, value):
        return [self._serialize_s(s) for s in value]

    def _serialize_ns(self, value):
        return [self._serialize_n(n) for n in value]

    def _serialize_bs(self, value):
        return [self._serialize_b(b) for b in value]

    def _serialize_l(self, value):
        return [self.serialize(v) for v in value]

    def _serialize_m(self, value):
        return dict([(k, self.serialize(v)) for k, v in value.items()])