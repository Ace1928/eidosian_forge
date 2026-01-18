import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
def _generate_multipart_form(self, parts):
    """Generate a multipart/form-data message.

        This is very loosely based on the email module in the Python standard
        library.  However, that module doesn't really support directly embedding
        binary data in a form: various versions of Python have mangled line
        separators in different ways, and none of them get it quite right.
        Since we only need a tiny subset of MIME here, it's easier to implement
        it ourselves.

        :return: a tuple of two elements: the Content-Type of the message, and
            the entire encoded message as a byte string.
        """
    encoded_parts = []
    for is_binary, name, value in parts:
        buf = io.BytesIO()
        if is_binary:
            ctype = 'application/octet-stream'
            cdisp = 'form-data; name="%s"; filename="%s"' % (quote(name), quote(name))
        else:
            ctype = 'text/plain; charset="utf-8"'
            cdisp = 'form-data; name="%s"' % quote(name)
        self._write_headers(buf, [('MIME-Version', '1.0'), ('Content-Type', ctype), ('Content-Disposition', cdisp)])
        if is_binary:
            if not isinstance(value, bytes):
                raise TypeError('bytes payload expected: %s' % type(value))
            buf.write(value)
        else:
            if not isinstance(value, _string_types):
                raise TypeError('string payload expected: %s' % type(value))
            lines = re.split('\\r\\n|\\r|\\n', value)
            for line in lines[:-1]:
                buf.write(line.encode('UTF-8'))
                buf.write(b'\r\n')
            buf.write(lines[-1].encode('UTF-8'))
        encoded_parts.append(buf.getvalue())
    boundary = self._make_boundary(b'\r\n'.join(encoded_parts))
    buf = io.BytesIO()
    ctype = 'multipart/form-data; boundary="%s"' % quote(boundary)
    self._write_headers(buf, [('MIME-Version', '1.0'), ('Content-Type', ctype)])
    for encoded_part in encoded_parts:
        self._write_boundary(buf, boundary)
        buf.write(encoded_part)
        buf.write(b'\r\n')
    self._write_boundary(buf, boundary, closing=True)
    return (ctype, buf.getvalue())