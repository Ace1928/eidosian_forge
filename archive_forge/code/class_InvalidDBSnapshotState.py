from boto.exception import JSONResponseError
class InvalidDBSnapshotState(JSONResponseError):
    pass