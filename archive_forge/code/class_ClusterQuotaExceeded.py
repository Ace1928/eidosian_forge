from boto.exception import JSONResponseError
class ClusterQuotaExceeded(JSONResponseError):
    pass