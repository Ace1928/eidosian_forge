from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import metrics
import six
def SnakeToCamelCase(value):
    """Convert value from snake_case to camelCase."""
    if not re.match('[a-zA-Z]+_[a-zA-Z]+', value):
        return value
    string = re.sub('^[\\-_\\.]', '', six.text_type(value.lower()))
    if not string:
        return string
    metrics.CustomTimedEvent(CAMEL_CASE_CONVERSION_EVENT)
    return string[0].lower() + re.sub('[\\-_\\.\\s]([a-z])', lambda matched: matched.group(1).upper(), string[1:])