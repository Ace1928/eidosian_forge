from boto.exception import BotoServerError
class InstrumentExpired(ResponseError):
    """The prepaid or the postpaid instrument has expired.
    """