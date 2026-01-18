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
def _needs_retry(self, attempts, operation_model, request_dict, response=None, caught_exception=None):
    service_id = operation_model.service_model.service_id.hyphenize()
    event_name = f'needs-retry.{service_id}.{operation_model.name}'
    responses = self._event_emitter.emit(event_name, response=response, endpoint=self, operation=operation_model, attempts=attempts, caught_exception=caught_exception, request_dict=request_dict)
    handler_response = first_non_none_response(responses)
    if handler_response is None:
        return False
    else:
        logger.debug('Response received to retry, sleeping for %s seconds', handler_response)
        time.sleep(handler_response)
        return True