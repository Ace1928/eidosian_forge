from boto.exception import JSONResponseError
class ClusterParameterGroupAlreadyExistsFault(JSONResponseError):
    pass