from boto.exception import JSONResponseError
class ReservedNodeNotFound(JSONResponseError):
    pass