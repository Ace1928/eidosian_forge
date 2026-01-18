from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
def _on_list_ready(self, parent, markers, operation, response):
    service_model = self.sqs_connection.meta.service_model
    if response.status == self.STATUS_CODE_OK:
        _, parsed = get_response(service_model.operation_model(operation), response.response)
        return parsed
    elif response.status in self.STATUS_CODES_TIMEOUT or response.status in self.STATUS_CODES_SERVER_ERRORS:
        return []
    else:
        raise self._for_status(response, response.read())