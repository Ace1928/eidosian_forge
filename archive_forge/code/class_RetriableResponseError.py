from boto.exception import BotoServerError
class RetriableResponseError(ResponseError):
    retry = True