from boto.exception import BotoServerError
class InsufficientS3BucketPolicyException(BotoServerError):
    """
    Raised when the S3 bucket does not allow Cloudtrail to
    write files into the prefix.
    """
    pass