import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
class RResponse:
    """
    A Response class that behaves in the way that mechanize expects it.
    Links to a requests.Response
    """

    def __init__(self, resp):
        self._resp = resp
        self.index = 0
        self.text = resp.text
        self._len = len(self.text)
        self.url = str(resp.url)
        self.statuscode = resp.status_code

    def geturl(self):
        return self._resp.url

    def __getitem__(self, item):
        try:
            return getattr(self._resp, item)
        except AttributeError:
            return getattr(self._resp.headers, item)

    def __getattribute__(self, item):
        try:
            return getattr(self._resp, item)
        except AttributeError:
            return getattr(self._resp.headers, item)

    def read(self, size=0):
        """
        Read from the content of the response. The class remembers what has
        been read so it's possible to read small consecutive parts of the
        content.

        :param size: The number of bytes to read
        :return: Somewhere between zero and 'size' number of bytes depending
            on how much it left in the content buffer to read.
        """
        if size:
            if self._len < size:
                return self.text
            else:
                if self._len == self.index:
                    part = None
                elif self._len - self.index < size:
                    part = self.text[self.index:]
                    self.index = self._len
                else:
                    part = self.text[self.index:self.index + size]
                    self.index += size
                return part
        else:
            return self.text