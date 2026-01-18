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
def _do_get_response(self, request, operation_model, context):
    try:
        logger.debug('Sending http request: %s', request)
        history_recorder.record('HTTP_REQUEST', {'method': request.method, 'headers': request.headers, 'streaming': operation_model.has_streaming_input, 'url': request.url, 'body': request.body})
        service_id = operation_model.service_model.service_id.hyphenize()
        event_name = f'before-send.{service_id}.{operation_model.name}'
        responses = self._event_emitter.emit(event_name, request=request)
        http_response = first_non_none_response(responses)
        if http_response is None:
            http_response = self._send(request)
    except HTTPClientError as e:
        return (None, e)
    except Exception as e:
        logger.debug('Exception received when sending HTTP request.', exc_info=True)
        return (None, e)
    response_dict = convert_to_response_dict(http_response, operation_model)
    handle_checksum_body(http_response, response_dict, context, operation_model)
    http_response_record_dict = response_dict.copy()
    http_response_record_dict['streaming'] = operation_model.has_streaming_output
    history_recorder.record('HTTP_RESPONSE', http_response_record_dict)
    protocol = operation_model.metadata['protocol']
    parser = self._response_parser_factory.create_parser(protocol)
    parsed_response = parser.parse(response_dict, operation_model.output_shape)
    if http_response.status_code >= 300:
        self._add_modeled_error_fields(response_dict, parsed_response, operation_model, parser)
    history_recorder.record('PARSED_RESPONSE', parsed_response)
    return ((http_response, parsed_response), None)