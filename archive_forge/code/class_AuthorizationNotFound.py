from boto.exception import JSONResponseError
class AuthorizationNotFound(JSONResponseError):
    pass