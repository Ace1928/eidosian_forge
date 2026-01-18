from boto.exception import JSONResponseError
class ClusterSnapshotNotFound(JSONResponseError):
    pass