from boto.exception import JSONResponseError
class DBSubnetQuotaExceeded(JSONResponseError):
    pass