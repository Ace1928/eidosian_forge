from decimal import Decimal, Context, Clamped
from decimal import Overflow, Inexact, Underflow, Rounded
from boto3.compat import collections_abc
from botocore.compat import six
def _deserialize_ns(self, value):
    return set(map(self._deserialize_n, value))