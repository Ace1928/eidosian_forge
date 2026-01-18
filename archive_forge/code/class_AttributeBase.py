from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
class AttributeBase(object):

    def __init__(self, name):
        self.name = name

    def __and__(self, value):
        raise DynamoDBOperationNotSupportedError('AND', self)

    def __or__(self, value):
        raise DynamoDBOperationNotSupportedError('OR', self)

    def __invert__(self):
        raise DynamoDBOperationNotSupportedError('NOT', self)

    def eq(self, value):
        """Creates a condition where the attribute is equal to the value.

        :param value: The value that the attribute is equal to.
        """
        return Equals(self, value)

    def lt(self, value):
        """Creates a condition where the attribute is less than the value.

        :param value: The value that the attribute is less than.
        """
        return LessThan(self, value)

    def lte(self, value):
        """Creates a condition where the attribute is less than or equal to the
           value.

        :param value: The value that the attribute is less than or equal to.
        """
        return LessThanEquals(self, value)

    def gt(self, value):
        """Creates a condition where the attribute is greater than the value.

        :param value: The value that the attribute is greater than.
        """
        return GreaterThan(self, value)

    def gte(self, value):
        """Creates a condition where the attribute is greater than or equal to
           the value.

        :param value: The value that the attribute is greater than or equal to.
        """
        return GreaterThanEquals(self, value)

    def begins_with(self, value):
        """Creates a condition where the attribute begins with the value.

        :param value: The value that the attribute begins with.
        """
        return BeginsWith(self, value)

    def between(self, low_value, high_value):
        """Creates a condition where the attribute is greater than or equal
        to the low value and less than or equal to the high value.

        :param low_value: The value that the attribute is greater than or equal to.
        :param high_value: The value that the attribute is less than or equal to.
        """
        return Between(self, low_value, high_value)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)