import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
def __request(self, methodname, params):
    request = dumps(params, methodname, encoding=self.__encoding, allow_none=self.__allow_none).encode(self.__encoding, 'xmlcharrefreplace')
    response = self.__transport.request(self.__host, self.__handler, request, verbose=self.__verbose)
    if len(response) == 1:
        response = response[0]
    return response