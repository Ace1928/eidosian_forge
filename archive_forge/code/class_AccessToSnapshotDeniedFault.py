from boto.exception import JSONResponseError
class AccessToSnapshotDeniedFault(JSONResponseError):
    pass