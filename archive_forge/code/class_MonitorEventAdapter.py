import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
class MonitorEventAdapter:

    def __init__(self, time=time.time):
        """Adapts event emitter events to produce monitor events

        :type time: callable
        :param time: A callable that produces the current time
        """
        self._time = time

    def feed(self, emitter_event_name, emitter_payload):
        """Feed an event emitter event to generate a monitor event

        :type emitter_event_name: str
        :param emitter_event_name: The name of the event emitted

        :type emitter_payload: dict
        :param emitter_payload: The payload to associated to the event
            emitted

        :rtype: BaseMonitorEvent
        :returns: A monitor event based on the event emitter events
            fired
        """
        return self._get_handler(emitter_event_name)(**emitter_payload)

    def _get_handler(self, event_name):
        return getattr(self, '_handle_' + event_name.split('.')[0].replace('-', '_'))

    def _handle_before_parameter_build(self, model, context, **kwargs):
        context['current_api_call_event'] = APICallEvent(service=model.service_model.service_id, operation=model.wire_name, timestamp=self._get_current_time())

    def _handle_request_created(self, request, **kwargs):
        context = request.context
        new_attempt_event = context['current_api_call_event'].new_api_call_attempt(timestamp=self._get_current_time())
        new_attempt_event.request_headers = request.headers
        new_attempt_event.url = request.url
        context['current_api_call_attempt_event'] = new_attempt_event

    def _handle_response_received(self, parsed_response, context, exception, **kwargs):
        attempt_event = context.pop('current_api_call_attempt_event')
        attempt_event.latency = self._get_latency(attempt_event)
        if parsed_response is not None:
            attempt_event.http_status_code = parsed_response['ResponseMetadata']['HTTPStatusCode']
            attempt_event.response_headers = parsed_response['ResponseMetadata']['HTTPHeaders']
            attempt_event.parsed_error = parsed_response.get('Error')
        else:
            attempt_event.wire_exception = exception
        return attempt_event

    def _handle_after_call(self, context, parsed, **kwargs):
        context['current_api_call_event'].retries_exceeded = parsed['ResponseMetadata'].get('MaxAttemptsReached', False)
        return self._complete_api_call(context)

    def _handle_after_call_error(self, context, exception, **kwargs):
        context['current_api_call_event'].retries_exceeded = self._is_retryable_exception(exception)
        return self._complete_api_call(context)

    def _is_retryable_exception(self, exception):
        return isinstance(exception, tuple(RETRYABLE_EXCEPTIONS['GENERAL_CONNECTION_ERROR']))

    def _complete_api_call(self, context):
        call_event = context.pop('current_api_call_event')
        call_event.latency = self._get_latency(call_event)
        return call_event

    def _get_latency(self, event):
        return self._get_current_time() - event.timestamp

    def _get_current_time(self):
        return int(self._time() * 1000)