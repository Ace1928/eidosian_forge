from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class ChecksumError(BotoCoreError):
    """The expected checksum did not match the calculated checksum."""
    fmt = 'Checksum {checksum_type} failed, expected checksum {expected_checksum} did not match calculated checksum {actual_checksum}.'