import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class NttCisResponse(XmlResponse):

    def parse_error(self):
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError(self.body)
        elif self.status == httplib.FORBIDDEN:
            raise InvalidCredsError(self.body)
        body = self.parse_body()
        if self.status == httplib.BAD_REQUEST:
            for response_code in BAD_CODE_XML_ELEMENTS:
                code = findtext(body, response_code[0], response_code[1])
                if code is not None:
                    break
            for message in BAD_MESSAGE_XML_ELEMENTS:
                message = findtext(body, message[0], message[1])
                if message is not None:
                    break
            raise NttCisAPIException(code=code, msg=message, driver=self.connection.driver)
        if self.status is not httplib.OK:
            raise NttCisAPIException(code=self.status, msg=body, driver=self.connection.driver)
        return self.body