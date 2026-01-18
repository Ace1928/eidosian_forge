import errno
import os
import random
import re
import socket
import time
from hashlib import md5
import six.moves.http_client as httplib
from six.moves import urllib as urlparse
from boto import config, UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from boto.s3.keyfile import KeyFile
def _upload_file_bytes(self, conn, http_conn, fp, file_length, total_bytes_uploaded, cb, num_cb, headers):
    """
        Makes one attempt to upload file bytes, using an existing resumable
        upload connection.

        Returns (etag, generation, metageneration) from server upon success.

        Raises ResumableUploadException if any problems occur.
        """
    buf = fp.read(self.BUFFER_SIZE)
    if cb:
        if num_cb > 2:
            cb_count = file_length / self.BUFFER_SIZE / (num_cb - 2)
        elif num_cb < 0:
            cb_count = -1
        else:
            cb_count = 0
        i = 0
        cb(total_bytes_uploaded, file_length)
    if not headers:
        put_headers = {}
    else:
        put_headers = headers.copy()
    if file_length:
        if total_bytes_uploaded == file_length:
            range_header = self._build_content_range_header('*', file_length)
        else:
            range_header = self._build_content_range_header('%d-%d' % (total_bytes_uploaded, file_length - 1), file_length)
        put_headers['Content-Range'] = range_header
    put_headers['Content-Length'] = str(file_length - total_bytes_uploaded)
    http_request = AWSAuthConnection.build_base_http_request(conn, 'PUT', path=self.tracker_uri_path, auth_path=None, headers=put_headers, host=self.tracker_uri_host)
    http_conn.putrequest('PUT', http_request.path)
    for k in put_headers:
        http_conn.putheader(k, put_headers[k])
    http_conn.endheaders()
    http_conn.set_debuglevel(0)
    while buf:
        http_conn.send(buf)
        for alg in self.digesters:
            self.digesters[alg].update(buf)
        total_bytes_uploaded += len(buf)
        if cb:
            i += 1
            if i == cb_count or cb_count == -1:
                cb(total_bytes_uploaded, file_length)
                i = 0
        buf = fp.read(self.BUFFER_SIZE)
    http_conn.set_debuglevel(conn.debug)
    if cb:
        cb(total_bytes_uploaded, file_length)
    if total_bytes_uploaded != file_length:
        raise ResumableUploadException('File changed during upload: EOF at %d bytes of %d byte file.' % (total_bytes_uploaded, file_length), ResumableTransferDisposition.ABORT)
    resp = http_conn.getresponse()
    http_conn.set_debuglevel(conn.debug)
    if resp.status == 200:
        return (resp.getheader('etag'), resp.getheader('x-goog-generation'), resp.getheader('x-goog-metageneration'))
    elif resp.status in [408, 500, 503]:
        disposition = ResumableTransferDisposition.WAIT_BEFORE_RETRY
    else:
        disposition = ResumableTransferDisposition.ABORT
    raise ResumableUploadException('Got response code %d while attempting upload (%s)' % (resp.status, resp.reason), disposition)