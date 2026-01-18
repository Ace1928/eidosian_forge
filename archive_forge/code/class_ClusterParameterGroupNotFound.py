from boto.exception import JSONResponseError
class ClusterParameterGroupNotFound(JSONResponseError):
    pass