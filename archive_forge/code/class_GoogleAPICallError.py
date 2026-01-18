from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class GoogleAPICallError(GoogleAPIError, metaclass=_GoogleAPICallErrorMeta):
    """Base class for exceptions raised by calling API methods.

    Args:
        message (str): The exception message.
        errors (Sequence[Any]): An optional list of error details.
        details (Sequence[Any]): An optional list of objects defined in google.rpc.error_details.
        response (Union[requests.Request, grpc.Call]): The response or
            gRPC call metadata.
        error_info (Union[error_details_pb2.ErrorInfo, None]): An optional object containing error info
            (google.rpc.error_details.ErrorInfo).
    """
    code: Union[int, None] = None
    'Optional[int]: The HTTP status code associated with this error.\n\n    This may be ``None`` if the exception does not have a direct mapping\n    to an HTTP error.\n\n    See http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html\n    '
    grpc_status_code = None
    'Optional[grpc.StatusCode]: The gRPC status code associated with this\n    error.\n\n    This may be ``None`` if the exception does not match up to a gRPC error.\n    '

    def __init__(self, message, errors=(), details=(), response=None, error_info=None):
        super(GoogleAPICallError, self).__init__(message)
        self.message = message
        'str: The exception message.'
        self._errors = errors
        self._details = details
        self._response = response
        self._error_info = error_info

    def __str__(self):
        error_msg = '{} {}'.format(self.code, self.message)
        if self.details:
            error_msg = '{} {}'.format(error_msg, self.details)
        elif self.errors:
            errors = [f'{error.code}: {error.message}' for error in self.errors if hasattr(error, 'code') and hasattr(error, 'message')]
            if errors:
                error_msg = '{} {}'.format(error_msg, '\n'.join(errors))
        return error_msg

    @property
    def reason(self):
        """The reason of the error.

        Reference:
            https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto#L112

        Returns:
            Union[str, None]: An optional string containing reason of the error.
        """
        return self._error_info.reason if self._error_info else None

    @property
    def domain(self):
        """The logical grouping to which the "reason" belongs.

        Reference:
            https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto#L112

        Returns:
            Union[str, None]: An optional string containing a logical grouping to which the "reason" belongs.
        """
        return self._error_info.domain if self._error_info else None

    @property
    def metadata(self):
        """Additional structured details about this error.

        Reference:
            https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto#L112

        Returns:
            Union[Dict[str, str], None]: An optional object containing structured details about the error.
        """
        return self._error_info.metadata if self._error_info else None

    @property
    def errors(self):
        """Detailed error information.

        Returns:
            Sequence[Any]: A list of additional error details.
        """
        return list(self._errors)

    @property
    def details(self):
        """Information contained in google.rpc.status.details.

        Reference:
            https://github.com/googleapis/googleapis/blob/master/google/rpc/status.proto
            https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto

        Returns:
            Sequence[Any]: A list of structured objects from error_details.proto
        """
        return list(self._details)

    @property
    def response(self):
        """Optional[Union[requests.Request, grpc.Call]]: The response or
        gRPC call metadata."""
        return self._response