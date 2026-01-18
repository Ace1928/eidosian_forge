from boto.exception import JSONResponseError
class DBSecurityGroupAlreadyExists(JSONResponseError):
    pass