from boto.exception import JSONResponseError
class PointInTimeRestoreNotEnabled(JSONResponseError):
    pass