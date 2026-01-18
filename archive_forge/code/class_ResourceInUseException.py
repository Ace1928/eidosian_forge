from boto.exception import JSONResponseError
class ResourceInUseException(JSONResponseError):
    pass