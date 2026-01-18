from __future__ import print_function
import email.generator as email_generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import io
import json
import mimetypes
import os
import threading
import six
from six.moves import http_client
from apitools.base.py import buffered_stream
from apitools.base.py import compression
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import stream_slice
from apitools.base.py import util
def __ConfigureMultipartRequest(self, http_request):
    """Configure http_request as a multipart request for this upload."""
    msg_root = mime_multipart.MIMEMultipart('related')
    setattr(msg_root, '_write_headers', lambda self: None)
    msg = mime_nonmultipart.MIMENonMultipart(*http_request.headers['content-type'].split('/'))
    msg.set_payload(http_request.body)
    msg_root.attach(msg)
    msg = mime_nonmultipart.MIMENonMultipart(*self.mime_type.split('/'))
    msg['Content-Transfer-Encoding'] = 'binary'
    msg.set_payload(self.stream.read())
    msg_root.attach(msg)
    fp = six.BytesIO()
    if six.PY3:
        generator_class = MultipartBytesGenerator
    else:
        generator_class = email_generator.Generator
    g = generator_class(fp, mangle_from_=False)
    g.flatten(msg_root, unixfrom=False)
    http_request.body = fp.getvalue()
    multipart_boundary = msg_root.get_boundary()
    http_request.headers['content-type'] = 'multipart/related; boundary=%r' % multipart_boundary
    if isinstance(multipart_boundary, six.text_type):
        multipart_boundary = multipart_boundary.encode('ascii')
    body_components = http_request.body.split(multipart_boundary)
    headers, _, _ = body_components[-2].partition(b'\n\n')
    body_components[-2] = b'\n\n'.join([headers, b'<media body>\n\n--'])
    http_request.loggable_body = multipart_boundary.join(body_components)