from boto.exception import BotoServerError, BotoClientError
from boto.exception import DynamoDBResponseError
class DynamoDBConditionalCheckFailedError(DynamoDBResponseError):
    """
    Raised when a ConditionalCheckFailedException response is received.
    This happens when a conditional check, expressed via the expected_value
    paramenter, fails.
    """
    pass