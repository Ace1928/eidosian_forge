from boto.exception import JSONResponseError
class DBParameterGroupAlreadyExists(JSONResponseError):
    pass