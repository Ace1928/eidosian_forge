from boto.exception import JSONResponseError
class ClusterAlreadyExists(JSONResponseError):
    pass