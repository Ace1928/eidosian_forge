from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
class Not(ConditionBase):
    expression_operator = 'NOT'
    expression_format = '({operator} {0})'