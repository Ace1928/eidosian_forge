from boto.exception import JSONResponseError
class AuthorizationAlreadyExistsFault(JSONResponseError):
    pass