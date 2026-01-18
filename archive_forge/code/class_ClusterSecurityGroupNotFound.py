from boto.exception import JSONResponseError
class ClusterSecurityGroupNotFound(JSONResponseError):
    pass