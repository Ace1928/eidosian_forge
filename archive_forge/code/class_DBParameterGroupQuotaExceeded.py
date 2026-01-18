from boto.exception import JSONResponseError
class DBParameterGroupQuotaExceeded(JSONResponseError):
    pass