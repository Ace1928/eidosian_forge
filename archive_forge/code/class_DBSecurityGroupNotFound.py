from boto.exception import JSONResponseError
class DBSecurityGroupNotFound(JSONResponseError):
    pass