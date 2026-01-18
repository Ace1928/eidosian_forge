from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def _get_name_placeholder(self):
    return '#' + self._name_placeholder + str(self._name_count)