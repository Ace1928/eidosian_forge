from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _deserialize_n(self, value):
    return DYNAMODB_CONTEXT.create_decimal(value)