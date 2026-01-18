from boto.exception import JSONResponseError
class ConditionalCheckFailedException(JSONResponseError):
    pass