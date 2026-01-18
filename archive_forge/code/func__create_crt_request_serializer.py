import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def _create_crt_request_serializer(session, region_name):
    return BotocoreCRTRequestSerializer(session, {'region_name': region_name, 'endpoint_url': None})