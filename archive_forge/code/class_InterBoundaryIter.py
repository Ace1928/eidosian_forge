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
class InterBoundaryIter:
    """
    A Producer that will iterate over boundaries.
    """

    def __init__(self, stream, boundary):
        self._stream = stream
        self._boundary = boundary

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return LazyStream(BoundaryIter(self._stream, self._boundary))
        except InputStreamExhausted:
            raise StopIteration()