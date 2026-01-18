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