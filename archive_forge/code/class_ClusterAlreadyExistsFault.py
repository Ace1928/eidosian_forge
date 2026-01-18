from boto.exception import JSONResponseError
class ClusterAlreadyExistsFault(JSONResponseError):
    pass