from boto.exception import BotoServerError
class SignatureDoesNotMatch(ResponseError):
    """The request signature calculated by Amazon does not match the
       signature you provided.
    """