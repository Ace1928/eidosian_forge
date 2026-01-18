from boto.exception import JSONResponseError
class ClusterQuotaExceededFault(JSONResponseError):
    pass