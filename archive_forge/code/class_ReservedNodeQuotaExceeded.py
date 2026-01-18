from boto.exception import JSONResponseError
class ReservedNodeQuotaExceeded(JSONResponseError):
    pass