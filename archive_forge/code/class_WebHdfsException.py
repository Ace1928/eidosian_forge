import io
import logging
import urllib.parse
from smart_open import utils, constants
import http.client as httplib
class WebHdfsException(Exception):

    def __init__(self, msg='', status_code=None):
        self.msg = msg
        self.status_code = status_code
        super(WebHdfsException, self).__init__(repr(self))

    def __repr__(self):
        return '{}(status_code={}, msg={!r})'.format(self.__class__.__name__, self.status_code, self.msg)

    @classmethod
    def from_response(cls, response):
        return cls(msg=response.text, status_code=response.status_code)