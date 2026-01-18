from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def _get_value_placeholder(self):
    return ':' + self._value_placeholder + str(self._value_count)