import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
class NonBooleanDynamizer(Dynamizer):
    """Casting boolean type to numeric types.

    This class is provided for backward compatibility.
    """

    def _get_dynamodb_type(self, attr):
        return get_dynamodb_type(attr, use_boolean=False)