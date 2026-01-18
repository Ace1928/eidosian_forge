from boto.exception import BotoServerError
class InsufficientSnsTopicPolicyException(BotoServerError):
    """
    Raised when the SNS topic does not allow Cloudtrail to post
    messages.
    """
    pass