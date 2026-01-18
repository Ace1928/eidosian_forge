from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
class BeginsWith(ConditionBase):
    expression_operator = 'begins_with'
    expression_format = '{operator}({0}, {1})'