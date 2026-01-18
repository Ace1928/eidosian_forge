from boto.exception import JSONResponseError
class InvalidClusterStateFault(JSONResponseError):
    pass