from boto.exception import JSONResponseError
class AuthorizationNotFoundFault(JSONResponseError):
    pass