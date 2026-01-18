import os
import copy
import hmac
import time
import base64
from hashlib import sha256
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import ET, b, httplib, urlparse, urlencode, basestring
from libcloud.utils.xml import fixxpath
from libcloud.common.base import (
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.common.azure_arm import AzureAuthJsonResponse, publicEnvironments
class AzureResponse(XmlResponse):
    valid_response_codes = [httplib.NOT_FOUND, httplib.CONFLICT, httplib.BAD_REQUEST, httplib.TEMPORARY_REDIRECT, httplib.PARTIAL_CONTENT]

    def success(self):
        i = int(self.status)
        return 200 <= i <= 299 or i in self.valid_response_codes

    def parse_error(self, msg=None):
        error_msg = 'Unknown error'
        try:
            body = self.parse_body()
            if type(body) == ET.Element:
                code = body.findtext(fixxpath(xpath='Code'))
                message = body.findtext(fixxpath(xpath='Message'))
                message = message.split('\n')[0]
                error_msg = '{}: {}'.format(code, message)
        except MalformedResponseError:
            pass
        if msg:
            error_msg = '{} - {}'.format(msg, error_msg)
        if self.status in [httplib.UNAUTHORIZED, httplib.FORBIDDEN]:
            raise InvalidCredsError(error_msg)
        raise LibcloudError('%s Status code: %d.' % (error_msg, self.status), driver=self)

    def parse_body(self):
        is_redirect = int(self.status) == httplib.TEMPORARY_REDIRECT
        if is_redirect and self.connection.driver.follow_redirects:
            raise AzureRedirectException(self)
        else:
            return super().parse_body()