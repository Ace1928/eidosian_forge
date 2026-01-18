import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
class BodyPart(object):
    """

    The ``BodyPart`` object is a ``Response``-like interface to an individual
    subpart of a multipart response. It is expected that these will
    generally be created by objects of the ``MultipartDecoder`` class.

    Like ``Response``, there is a ``CaseInsensitiveDict`` object named headers,
    ``content`` to access bytes, ``text`` to access unicode, and ``encoding``
    to access the unicode codec.

    """

    def __init__(self, content, encoding):
        self.encoding = encoding
        headers = {}
        if b'\r\n\r\n' in content:
            first, self.content = _split_on_find(content, b'\r\n\r\n')
            if first != b'':
                headers = _header_parser(first.lstrip(), encoding)
        else:
            raise ImproperBodyPartContentException('content does not contain CR-LF-CR-LF')
        self.headers = CaseInsensitiveDict(headers)

    @property
    def text(self):
        """Content of the ``BodyPart`` in unicode."""
        return self.content.decode(self.encoding)