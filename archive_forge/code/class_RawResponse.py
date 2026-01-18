import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
class RawResponse(Response):

    def __init__(self, connection, response=None):
        """
        :param connection: Parent connection object.
        :type connection: :class:`.Connection`
        """
        self._status = None
        self._response = None
        self._headers = {}
        self._error = None
        self._reason = None
        self.connection = connection
        if response is not None:
            self.headers = lowercase_keys(dict(response.headers))
            self.error = response.reason
            self.status = response.status_code
            self.request = response.request
            self.iter_content = response.iter_content
        if not self.success():
            self.parse_error()

    def success(self):
        """
        Determine if our request was successful.

        The meaning of this can be arbitrary; did we receive OK status? Did
        the node get created? Were we authenticated?

        :rtype: ``bool``
        :return: ``True`` or ``False``
        """
        import requests
        return self.status in [requests.codes.ok, requests.codes.created, httplib.OK, httplib.CREATED, httplib.ACCEPTED]

    @property
    def response(self):
        if not self._response:
            response = self.connection.connection.getresponse()
            self._response = HttpLibResponseProxy(response)
            if not self.success():
                self.parse_error()
        return self._response

    @property
    def body(self):
        return self.response.body

    @property
    def reason(self):
        if not self._reason:
            self._reason = self.response.reason
        return self._reason