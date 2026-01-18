from boto.exception import JSONResponseError
class ResizeNotFoundFault(JSONResponseError):
    pass