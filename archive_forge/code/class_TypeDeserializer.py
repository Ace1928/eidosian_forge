from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
class TypeDeserializer(object):
    """This class deserializes DynamoDB types to Python types."""

    def deserialize(self, value):
        """The method to deserialize the DynamoDB data types.

        :param value: A DynamoDB value to be deserialized to a pythonic value.
            Here are the various conversions:

            DynamoDB                                Python
            --------                                ------
            {'NULL': True}                          None
            {'BOOL': True/False}                    True/False
            {'N': str(value)}                       Decimal(str(value))
            {'S': string}                           string
            {'B': bytes}                            Binary(bytes)
            {'NS': [str(value)]}                    set([Decimal(str(value))])
            {'SS': [string]}                        set([string])
            {'BS': [bytes]}                         set([bytes])
            {'L': list}                             list
            {'M': dict}                             dict

        :returns: The pythonic value of the DynamoDB type.
        """
        if not value:
            raise TypeError('Value must be a nonempty dictionary whose key is a valid dynamodb type.')
        dynamodb_type = list(value.keys())[0]
        try:
            deserializer = getattr(self, '_deserialize_%s' % dynamodb_type.lower())
        except AttributeError:
            raise TypeError('Dynamodb type %s is not supported' % dynamodb_type)
        return deserializer(value[dynamodb_type])

    def _deserialize_null(self, value):
        return None

    def _deserialize_bool(self, value):
        return value

    def _deserialize_n(self, value):
        return DYNAMODB_CONTEXT.create_decimal(value)

    def _deserialize_s(self, value):
        return value

    def _deserialize_b(self, value):
        return Binary(value)

    def _deserialize_ns(self, value):
        return set(map(self._deserialize_n, value))

    def _deserialize_ss(self, value):
        return set(map(self._deserialize_s, value))

    def _deserialize_bs(self, value):
        return set(map(self._deserialize_b, value))

    def _deserialize_l(self, value):
        return [self.deserialize(v) for v in value]

    def _deserialize_m(self, value):
        return dict([(k, self.deserialize(v)) for k, v in value.items()])