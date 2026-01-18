from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def _build_value_placeholder(self, value, attribute_value_placeholders, has_grouped_values=False):
    if has_grouped_values:
        placeholder_list = []
        for v in value:
            value_placeholder = self._get_value_placeholder()
            self._value_count += 1
            placeholder_list.append(value_placeholder)
            attribute_value_placeholders[value_placeholder] = v
        return '(' + ', '.join(placeholder_list) + ')'
    else:
        value_placeholder = self._get_value_placeholder()
        self._value_count += 1
        attribute_value_placeholders[value_placeholder] = value
        return value_placeholder