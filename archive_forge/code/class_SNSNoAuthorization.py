from boto.exception import JSONResponseError
class SNSNoAuthorization(JSONResponseError):
    pass