from boto.exception import JSONResponseError
class UnsupportedOptionFault(JSONResponseError):
    pass