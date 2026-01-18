import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def _create_crt_s3_client(session, config, region_name, credentials, lock, **kwargs):
    """Create boto3 wrapper class to manage crt lock reference and S3 client."""
    cred_wrapper = BotocoreCRTCredentialsWrapper(credentials)
    cred_provider = cred_wrapper.to_crt_credentials_provider()
    return CRTS3Client(_create_crt_client(session, config, region_name, cred_provider), lock, region_name, cred_wrapper)