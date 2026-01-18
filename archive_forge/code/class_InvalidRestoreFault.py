from boto.exception import JSONResponseError
class InvalidRestoreFault(JSONResponseError):
    pass