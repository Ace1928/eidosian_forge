from boto.exception import JSONResponseError
class ReservedDBInstanceQuotaExceeded(JSONResponseError):
    pass