from boto.exception import JSONResponseError
class ClusterSnapshotAlreadyExistsFault(JSONResponseError):
    pass