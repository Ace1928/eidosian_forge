from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnsupportedS3ControlArnError(BotoCoreError):
    """Error when S3 ARN provided to S3 control parameter is not supported"""
    fmt = 'S3 ARN "{arn}" provided is invalid for this operation. {msg}'