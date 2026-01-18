import datetime
import logging
import os
import threading
import time
import uuid
from botocore import parsers
from botocore.awsrequest import create_request_object
from botocore.exceptions import HTTPClientError
from botocore.history import get_global_history_recorder
from botocore.hooks import first_non_none_response
from botocore.httpchecksum import handle_checksum_body
from botocore.httpsession import URLLib3Session
from botocore.response import StreamingBody
from botocore.utils import (
def _calculate_ttl(self, response_received_timestamp, date_header, read_timeout):
    local_timestamp = datetime.datetime.utcnow()
    date_conversion = datetime.datetime.strptime(date_header, '%a, %d %b %Y %H:%M:%S %Z')
    estimated_skew = date_conversion - response_received_timestamp
    ttl = local_timestamp + datetime.timedelta(seconds=read_timeout) + estimated_skew
    return ttl.strftime('%Y%m%dT%H%M%SZ')