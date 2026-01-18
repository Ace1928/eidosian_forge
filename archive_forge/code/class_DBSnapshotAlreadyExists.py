from boto.exception import JSONResponseError
class DBSnapshotAlreadyExists(JSONResponseError):
    pass