from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _is_number(self, value):
    if isinstance(value, (six.integer_types, Decimal)):
        return True
    elif isinstance(value, float):
        raise TypeError('Float types are not supported. Use Decimal types instead.')
    return False