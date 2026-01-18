from boto.exception import JSONResponseError
class ClusterSnapshotAlreadyExists(JSONResponseError):
    pass