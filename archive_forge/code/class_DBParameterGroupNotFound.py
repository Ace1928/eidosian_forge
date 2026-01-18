from boto.exception import JSONResponseError
class DBParameterGroupNotFound(JSONResponseError):
    pass