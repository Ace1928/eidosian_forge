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
def _attempt_resumable_upload(self, key, fp, file_length, headers, cb, num_cb):
    """
        Attempts a resumable upload.

        Returns (etag, generation, metageneration) from server upon success.

        Raises ResumableUploadException if any problems occur.
        """
    server_start, server_end = self.SERVER_HAS_NOTHING
    conn = key.bucket.connection
    if self.tracker_uri:
        try:
            server_start, server_end = self._query_server_pos(conn, file_length)
            self.server_has_bytes = server_start
            if server_end:
                print('Catching up hash digest(s) for resumed upload')
                fp.seek(0)
                bytes_to_go = server_end + 1
                while bytes_to_go:
                    chunk = fp.read(min(key.BufferSize, bytes_to_go))
                    if not chunk:
                        raise ResumableUploadException('Hit end of file during resumable upload hash catchup. This should not happen under\nnormal circumstances, as it indicates the server has more bytes of this transfer\nthan the current file size. Restarting upload.', ResumableTransferDisposition.START_OVER)
                    for alg in self.digesters:
                        self.digesters[alg].update(chunk)
                    bytes_to_go -= len(chunk)
            if conn.debug >= 1:
                print('Resuming transfer.')
        except ResumableUploadException as e:
            if conn.debug >= 1:
                print('Unable to resume transfer (%s).' % e.message)
            self._start_new_resumable_upload(key, headers)
    else:
        self._start_new_resumable_upload(key, headers)
    if self.upload_start_point is None:
        self.upload_start_point = server_end
    total_bytes_uploaded = server_end + 1
    if file_length < total_bytes_uploaded:
        fp.seek(total_bytes_uploaded)
    conn = key.bucket.connection
    http_conn = conn.new_http_connection(self.tracker_uri_host, conn.port, conn.is_secure)
    http_conn.set_debuglevel(conn.debug)
    try:
        return self._upload_file_bytes(conn, http_conn, fp, file_length, total_bytes_uploaded, cb, num_cb, headers)
    except (ResumableUploadException, socket.error):
        resp = self._query_server_state(conn, file_length)
        if resp.status == 400:
            raise ResumableUploadException('Got 400 response from server state query after failed resumable upload attempt. This can happen for various reasons, including specifying an invalid request (e.g., an invalid canned ACL) or if the file size changed between upload attempts', ResumableTransferDisposition.ABORT)
        else:
            raise
    finally:
        http_conn.close()