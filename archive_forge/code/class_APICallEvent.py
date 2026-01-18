import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
class APICallEvent(BaseMonitorEvent):

    def __init__(self, service, operation, timestamp, latency=None, attempts=None, retries_exceeded=False):
        """Monitor event for a single API call

        This event corresponds to a single client method call, which includes
        every HTTP requests attempt made in order to complete the client call

        :type service: str
        :param service: A string identifying the service associated to
            the event

        :type operation: str
        :param operation: A string identifying the operation of service
            associated to the event

        :type timestamp: int
        :param timestamp: Epoch time in milliseconds from when the event began

        :type latency: int
        :param latency: The time in milliseconds to complete the client call

        :type attempts: list
        :param attempts: The list of APICallAttempts associated to the
            APICall

        :type retries_exceeded: bool
        :param retries_exceeded: True if API call exceeded retries. False
            otherwise
        """
        super().__init__(service=service, operation=operation, timestamp=timestamp)
        self.latency = latency
        self.attempts = attempts
        if attempts is None:
            self.attempts = []
        self.retries_exceeded = retries_exceeded

    def new_api_call_attempt(self, timestamp):
        """Instantiates APICallAttemptEvent associated to the APICallEvent

        :type timestamp: int
        :param timestamp: Epoch time in milliseconds to associate to the
            APICallAttemptEvent
        """
        attempt_event = APICallAttemptEvent(service=self.service, operation=self.operation, timestamp=timestamp)
        self.attempts.append(attempt_event)
        return attempt_event