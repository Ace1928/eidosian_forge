from boto.exception import JSONResponseError
class CaseCreationLimitExceeded(JSONResponseError):
    pass