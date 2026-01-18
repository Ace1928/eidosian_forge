from boto.exception import JSONResponseError
class ClusterSnapshotQuotaExceededFault(JSONResponseError):
    pass