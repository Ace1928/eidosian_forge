from boto.exception import BotoServerError
class PrepaidFundingLimitExceeded(RetriableResponseError):
    """An attempt has been made to fund the prepaid instrument
       at a level greater than its recharge limit.
    """