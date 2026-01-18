from __future__ import absolute_import, division, unicode_literals
from six import text_type
from six.moves import http_client, urllib
import codecs
import re
from io import BytesIO, StringIO
from tensorboard._vendor import webencodings
from .constants import EOF, spaceCharacters, asciiLetters, asciiUppercase
from .constants import _ReparseException
from . import _utils
def _readStream(self, bytes):
    data = self.stream.read(bytes)
    self.buffer.append(data)
    self.position[0] += 1
    self.position[1] = len(data)
    return data