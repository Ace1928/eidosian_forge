import collections
import email.generator as generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import email.parser as email_parser
import itertools
import time
import uuid
import six
from six.moves import http_client
from six.moves import urllib_parse
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class BatchApiRequest(object):
    """Batches multiple api requests into a single request."""

    class ApiCall(object):
        """Holds request and response information for each request.

        ApiCalls are ultimately exposed to the client once the HTTP
        batch request has been completed.

        Attributes:
          http_request: A client-supplied http_wrapper.Request to be
              submitted to the server.
          response: A http_wrapper.Response object given by the server as a
              response to the user request, or None if an error occurred.
          exception: An apiclient.errors.HttpError object if an error
              occurred, or None.

        """

        def __init__(self, request, retryable_codes, service, method_config):
            """Initialize an individual API request.

            Args:
              request: An http_wrapper.Request object.
              retryable_codes: A list of integer HTTP codes that can
                  be retried.
              service: A service inheriting from base_api.BaseApiService.
              method_config: Method config for the desired API request.

            """
            self.__retryable_codes = list(set(retryable_codes + [http_client.UNAUTHORIZED]))
            self.__http_response = None
            self.__service = service
            self.__method_config = method_config
            self.http_request = request
            self.__response = None
            self.__exception = None

        @property
        def is_error(self):
            return self.exception is not None

        @property
        def response(self):
            return self.__response

        @property
        def exception(self):
            return self.__exception

        @property
        def authorization_failed(self):
            return self.__http_response and self.__http_response.status_code == http_client.UNAUTHORIZED

        @property
        def terminal_state(self):
            if self.__http_response is None:
                return False
            response_code = self.__http_response.status_code
            return response_code not in self.__retryable_codes

        def HandleResponse(self, http_response, exception):
            """Handles incoming http response to the request in http_request.

            This is intended to be used as a callback function for
            BatchHttpRequest.Add.

            Args:
              http_response: Deserialized http_wrapper.Response object.
              exception: apiclient.errors.HttpError object if an error
                  occurred.

            """
            self.__http_response = http_response
            self.__exception = exception
            if self.terminal_state and (not self.__exception):
                self.__response = self.__service.ProcessHttpResponse(self.__method_config, self.__http_response)

    def __init__(self, batch_url=None, retryable_codes=None, response_encoding=None):
        """Initialize a batch API request object.

        Args:
          batch_url: Base URL for batch API calls.
          retryable_codes: A list of integer HTTP codes that can be retried.
          response_encoding: The encoding type of response content.
        """
        self.api_requests = []
        self.retryable_codes = retryable_codes or []
        self.batch_url = batch_url or 'https://www.googleapis.com/batch'
        self.response_encoding = response_encoding

    def Add(self, service, method, request, global_params=None):
        """Add a request to the batch.

        Args:
          service: A class inheriting base_api.BaseApiService.
          method: A string indicated desired method from the service. See
              the example in the class docstring.
          request: An input message appropriate for the specified
              service.method.
          global_params: Optional additional parameters to pass into
              method.PrepareHttpRequest.

        Returns:
          None

        """
        method_config = service.GetMethodConfig(method)
        upload_config = service.GetUploadConfig(method)
        http_request = service.PrepareHttpRequest(method_config, request, global_params=global_params, upload_config=upload_config)
        api_request = self.ApiCall(http_request, self.retryable_codes, service, method_config)
        self.api_requests.append(api_request)

    def Execute(self, http, sleep_between_polls=5, max_retries=5, max_batch_size=None, batch_request_callback=None):
        """Execute all of the requests in the batch.

        Args:
          http: httplib2.Http object for use in the request.
          sleep_between_polls: Integer number of seconds to sleep between
              polls.
          max_retries: Max retries. Any requests that have not succeeded by
              this number of retries simply report the last response or
              exception, whatever it happened to be.
          max_batch_size: int, if specified requests will be split in batches
              of given size.
          batch_request_callback: function of (http_response, exception) passed
              to BatchHttpRequest which will be run on any given results.

        Returns:
          List of ApiCalls.
        """
        requests = [request for request in self.api_requests if not request.terminal_state]
        batch_size = max_batch_size or len(requests)
        for attempt in range(max_retries):
            if attempt:
                time.sleep(sleep_between_polls)
            for i in range(0, len(requests), batch_size):
                batch_http_request = BatchHttpRequest(batch_url=self.batch_url, callback=batch_request_callback, response_encoding=self.response_encoding)
                for request in itertools.islice(requests, i, i + batch_size):
                    batch_http_request.Add(request.http_request, request.HandleResponse)
                batch_http_request.Execute(http)
                if hasattr(http.request, 'credentials'):
                    if any((request.authorization_failed for request in itertools.islice(requests, i, i + batch_size))):
                        http.request.credentials.refresh(http)
            requests = [request for request in self.api_requests if not request.terminal_state]
            if not requests:
                break
        return self.api_requests