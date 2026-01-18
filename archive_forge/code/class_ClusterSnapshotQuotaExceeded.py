from boto.exception import JSONResponseError
class ClusterSnapshotQuotaExceeded(JSONResponseError):
    pass