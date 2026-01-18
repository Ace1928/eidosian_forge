import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def _create_crt_client(session, config, region_name, cred_provider):
    """Create a CRT S3 Client for file transfer.

    Instantiating many of these may lead to degraded performance or
    system resource exhaustion.
    """
    create_crt_client_kwargs = {'region': region_name, 'use_ssl': True, 'crt_credentials_provider': cred_provider}
    return create_s3_crt_client(**create_crt_client_kwargs)