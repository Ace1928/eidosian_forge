from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
def _get_retry_info(self, response):
    retry_info = ''
    if 'ResponseMetadata' in response:
        metadata = response['ResponseMetadata']
        if metadata.get('MaxAttemptsReached', False):
            if 'RetryAttempts' in metadata:
                retry_info = f' (reached max retries: {metadata['RetryAttempts']})'
    return retry_info