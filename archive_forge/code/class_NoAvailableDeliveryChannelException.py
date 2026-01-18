from boto.exception import BotoServerError
class NoAvailableDeliveryChannelException(BotoServerError):
    pass