from boto.exception import JSONResponseError
class ClusterParameterGroupAlreadyExists(JSONResponseError):
    pass