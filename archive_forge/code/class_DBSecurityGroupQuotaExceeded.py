from boto.exception import JSONResponseError
class DBSecurityGroupQuotaExceeded(JSONResponseError):
    pass