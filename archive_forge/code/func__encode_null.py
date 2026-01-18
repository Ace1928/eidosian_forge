import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def _encode_null(self, attr):
    return True