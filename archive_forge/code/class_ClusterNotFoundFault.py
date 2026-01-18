from boto.exception import JSONResponseError
class ClusterNotFoundFault(JSONResponseError):
    pass