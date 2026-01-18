from boto.exception import JSONResponseError
class NumberOfNodesQuotaExceeded(JSONResponseError):
    pass