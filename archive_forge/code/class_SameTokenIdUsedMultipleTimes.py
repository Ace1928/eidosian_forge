from boto.exception import BotoServerError
class SameTokenIdUsedMultipleTimes(ResponseError):
    """This token is already used in earlier transactions.
    """