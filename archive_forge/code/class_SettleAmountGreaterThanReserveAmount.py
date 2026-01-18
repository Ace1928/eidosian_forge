from boto.exception import BotoServerError
class SettleAmountGreaterThanReserveAmount(ResponseError):
    """The amount being settled is greater than the reserved amount.
    """