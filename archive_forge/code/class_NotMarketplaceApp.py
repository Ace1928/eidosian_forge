from boto.exception import BotoServerError
class NotMarketplaceApp(RetriableResponseError):
    """This is not an marketplace application or the caller does not
       match either the sender or the recipient.
    """