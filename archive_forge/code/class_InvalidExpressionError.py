from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidExpressionError(BotoCoreError):
    """Expression is either invalid or too complex."""
    fmt = 'Invalid expression {expression}: Only dotted lookups are supported.'