from boto.exception import JSONResponseError
class SnapshotCopyDisabled(JSONResponseError):
    pass