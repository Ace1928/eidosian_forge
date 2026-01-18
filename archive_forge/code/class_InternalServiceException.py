from boto.exception import JSONResponseError
class InternalServiceException(JSONResponseError):
    pass