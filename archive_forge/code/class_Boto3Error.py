import botocore.exceptions
class Boto3Error(Exception):
    """Base class for all Boto3 errors."""