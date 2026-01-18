from boto.exception import JSONResponseError
class SnapshotQuotaExceeded(JSONResponseError):
    pass