from boto.exception import BotoServerError, BotoClientError
from boto.exception import DynamoDBResponseError
class DynamoDBItemError(BotoClientError):
    """
    Raised when invalid parameters are passed when creating a
    new Item in DynamoDB.
    """
    pass