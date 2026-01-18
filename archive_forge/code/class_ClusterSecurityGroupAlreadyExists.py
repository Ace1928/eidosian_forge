from boto.exception import JSONResponseError
class ClusterSecurityGroupAlreadyExists(JSONResponseError):
    pass