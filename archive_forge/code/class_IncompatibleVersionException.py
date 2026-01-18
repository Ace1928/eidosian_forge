from boto.exception import JSONResponseError
class IncompatibleVersionException(JSONResponseError):
    pass