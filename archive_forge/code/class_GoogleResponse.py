import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
class GoogleResponse(JsonResponse):
    """
    Google Base Response class.
    """

    def success(self):
        """
        Determine if the request was successful.

        For the Google response class, tag all responses as successful and
        raise appropriate Exceptions from parse_body.

        :return: C{True}
        """
        return True

    def _get_error(self, body):
        """
        Get the error code and message from a JSON response.

        Return just the first error if there are multiple errors.

        :param  body: The body of the JSON response dictionary
        :type   body: ``dict``

        :return:  Tuple containing error code and message
        :rtype:   ``tuple`` of ``str`` or ``int``
        """
        if 'errors' in body['error']:
            err = body['error']['errors'][0]
        else:
            err = body['error']
        if 'code' in err:
            code = err.get('code')
            message = err.get('message')
        else:
            code = None
            if 'reason' in err:
                code = err.get('reason')
            message = body.get('error_description', err)
        return (code, message)

    def parse_body(self):
        """
        Parse the JSON response body, or raise exceptions as appropriate.

        :return:  JSON dictionary
        :rtype:   ``dict``
        """
        if len(self.body) == 0 and (not self.parse_zero_length_body):
            return self.body
        json_error = False
        try:
            body = json.loads(self.body)
        except Exception:
            body = self.body
            json_error = True
        valid_http_codes = [httplib.OK, httplib.CREATED, httplib.ACCEPTED, httplib.CONFLICT]
        if self.status in valid_http_codes:
            if json_error:
                raise JsonParseError(body, self.status, None)
            elif 'error' in body:
                code, message = self._get_error(body)
                if code == 'QUOTA_EXCEEDED':
                    raise QuotaExceededError(message, self.status, code)
                elif code == 'RESOURCE_ALREADY_EXISTS':
                    raise ResourceExistsError(message, self.status, code)
                elif code == 'alreadyExists':
                    raise ResourceExistsError(message, self.status, code)
                elif code.startswith('RESOURCE_IN_USE'):
                    raise ResourceInUseError(message, self.status, code)
                else:
                    raise GoogleBaseError(message, self.status, code)
            else:
                return body
        elif self.status == httplib.NOT_FOUND:
            if not json_error and 'error' in body:
                code, message = self._get_error(body)
            else:
                message = body
                code = None
            raise ResourceNotFoundError(message, self.status, code)
        elif self.status == httplib.BAD_REQUEST:
            if not json_error and 'error' in body:
                code, message = self._get_error(body)
            else:
                message = body
                code = None
            raise InvalidRequestError(message, self.status, code)
        else:
            if not json_error and 'error' in body:
                code, message = self._get_error(body)
            else:
                message = body
                code = None
            raise GoogleBaseError(message, self.status, code)