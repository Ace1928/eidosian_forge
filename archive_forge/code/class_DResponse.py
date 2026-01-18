import json
import logging
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
class DResponse:
    """A Response class that behaves in the way that mechanize expects it"""

    def __init__(self, **kwargs):
        self.status = 200
        self.index = 0
        self._message = ''
        self.url = ''
        if kwargs:
            for key, val in kwargs.items():
                if val:
                    self.__setitem__(key, val)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, item):
        if item == 'content-location':
            return self.url
        elif item == 'content-length':
            return len(self._message)
        else:
            return getattr(self, item)

    def geturl(self):
        """
        The base url for the response

        :return: The url
        """
        return self.url

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
                return self._message
            else:
                if self._len == self.index:
                    part = None
                elif self._len - self.index < size:
                    part = self._message[self.index:]
                    self.index = self._len
                else:
                    part = self._message[self.index:self.index + size]
                    self.index += size
                return part
        else:
            return self._message

    def write(self, message):
        """
        Write the message into the content buffer

        :param message: The message
        """
        self._message = message
        self._len = len(message)