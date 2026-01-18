from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
class Contains(ConditionBase):
    expression_operator = 'contains'
    expression_format = '{operator}({0}, {1})'