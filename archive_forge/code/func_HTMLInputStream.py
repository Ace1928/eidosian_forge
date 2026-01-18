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
def HTMLInputStream(source, **kwargs):
    if isinstance(source, http_client.HTTPResponse) or (isinstance(source, urllib.response.addbase) and isinstance(source.fp, http_client.HTTPResponse)):
        isUnicode = False
    elif hasattr(source, 'read'):
        isUnicode = isinstance(source.read(0), text_type)
    else:
        isUnicode = isinstance(source, text_type)
    if isUnicode:
        encodings = [x for x in kwargs if x.endswith('_encoding')]
        if encodings:
            raise TypeError('Cannot set an encoding with a unicode input, set %r' % encodings)
        return HTMLUnicodeInputStream(source, **kwargs)
    else:
        return HTMLBinaryInputStream(source, **kwargs)