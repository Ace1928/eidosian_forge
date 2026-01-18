from boto.exception import BotoServerError
class InstrumentAccessDenied(ResponseError):
    """The external calling application is not the recipient for this
       postpaid or prepaid instrument.
    """