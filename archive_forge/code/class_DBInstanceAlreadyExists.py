from boto.exception import JSONResponseError
class DBInstanceAlreadyExists(JSONResponseError):
    pass