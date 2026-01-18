from boto.exception import JSONResponseError
class InvalidClusterSnapshotStateFault(JSONResponseError):
    pass