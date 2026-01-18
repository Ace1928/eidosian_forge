import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
class APICallAttemptEvent(BaseMonitorEvent):

    def __init__(self, service, operation, timestamp, latency=None, url=None, http_status_code=None, request_headers=None, response_headers=None, parsed_error=None, wire_exception=None):
        """Monitor event for a single API call attempt

        This event corresponds to a single HTTP request attempt in completing
        the entire client method call.

        :type service: str
        :param service: A string identifying the service associated to
            the event

        :type operation: str
        :param operation: A string identifying the operation of service
            associated to the event

        :type timestamp: int
        :param timestamp: Epoch time in milliseconds from when the HTTP request
            started

        :type latency: int
        :param latency: The time in milliseconds to complete the HTTP request
            whether it succeeded or failed

        :type url: str
        :param url: The URL the attempt was sent to

        :type http_status_code: int
        :param http_status_code: The HTTP status code of the HTTP response
            if there was a response

        :type request_headers: dict
        :param request_headers: The HTTP headers sent in making the HTTP
            request

        :type response_headers: dict
        :param response_headers: The HTTP headers returned in the HTTP response
            if there was a response

        :type parsed_error: dict
        :param parsed_error: The error parsed if the service returned an
            error back

        :type wire_exception: Exception
        :param wire_exception: The exception raised in sending the HTTP
            request (i.e. ConnectionError)
        """
        super().__init__(service=service, operation=operation, timestamp=timestamp)
        self.latency = latency
        self.url = url
        self.http_status_code = http_status_code
        self.request_headers = request_headers
        self.response_headers = response_headers
        self.parsed_error = parsed_error
        self.wire_exception = wire_exception