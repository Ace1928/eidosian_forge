import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import (
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
class ChunkIter:
    """
    An iterable that will yield chunks of data. Given a file-like object as the
    constructor, yield chunks of read operations from that object.
    """

    def __init__(self, flo, chunk_size=64 * 1024):
        self.flo = flo
        self.chunk_size = chunk_size

    def __next__(self):
        try:
            data = self.flo.read(self.chunk_size)
        except InputStreamExhausted:
            raise StopIteration()
        if data:
            return data
        else:
            raise StopIteration()

    def __iter__(self):
        return self