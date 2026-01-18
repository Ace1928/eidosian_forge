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
def _set_ttl(self, retries_context, read_timeout, success_response):
    response_date_header = success_response[0].headers.get('Date')
    has_streaming_input = retries_context.get('has_streaming_input')
    if response_date_header and (not has_streaming_input):
        try:
            response_received_timestamp = datetime.datetime.utcnow()
            retries_context['ttl'] = self._calculate_ttl(response_received_timestamp, response_date_header, read_timeout)
        except Exception:
            logger.debug('Exception received when updating retries context with TTL', exc_info=True)