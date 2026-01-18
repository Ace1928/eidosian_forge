from boto.exception import BotoServerError
class InvalidSnsTopicNameException(BotoServerError):
    """
    Raised when an invalid SNS topic name is passed to Cloudtrail.
    """
    pass