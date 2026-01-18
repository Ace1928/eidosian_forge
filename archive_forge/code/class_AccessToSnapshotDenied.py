from boto.exception import JSONResponseError
class AccessToSnapshotDenied(JSONResponseError):
    pass