import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def _write_details(self, details):
    """Output details to the stream.

        :param details: An extended details dict for a test outcome.
        """
    self._stream.write(_b(' [ multipart\n'))
    for name, content in sorted(details.items()):
        self._stream.write(_b('Content-Type: %s/%s' % (content.content_type.type, content.content_type.subtype)))
        parameters = content.content_type.parameters
        if parameters:
            self._stream.write(_b(';'))
            param_strs = []
            for param, value in sorted(parameters.items()):
                param_strs.append('%s=%s' % (param, value))
            self._stream.write(_b(','.join(param_strs)))
        self._stream.write(_b('\n%s\n' % name))
        encoder = chunked.Encoder(self._stream)
        list(map(encoder.write, content.iter_bytes()))
        encoder.close()