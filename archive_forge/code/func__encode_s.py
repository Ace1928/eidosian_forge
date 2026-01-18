import base64
from decimal import (Decimal, DecimalException, Context,
from collections.abc import Mapping
from boto.dynamodb.exceptions import DynamoDBNumberError
from boto.compat import filter, map, six, long_type
def _encode_s(self, attr):
    if isinstance(attr, bytes):
        attr = attr.decode('utf-8')
    elif not isinstance(attr, six.text_type):
        attr = str(attr)
    return attr