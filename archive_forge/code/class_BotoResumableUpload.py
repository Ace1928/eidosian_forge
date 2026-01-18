from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import random
import re
import socket
import time
import six
from six.moves import urllib
from six.moves import http_client
from boto import config
from boto import UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from gslib.exception import InvalidUrlError
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.constants import XML_PROGRESS_CALLBACKS
from gslib.utils.constants import UTF8
class BotoResumableUpload(object):
    """Upload helper class for resumable uploads via boto."""
    BUFFER_SIZE = 8192
    RETRYABLE_EXCEPTIONS = (http_client.HTTPException, IOError, socket.error, socket.gaierror)
    SERVICE_HAS_NOTHING = (0, -1)

    def __init__(self, tracker_callback, logger, resume_url=None, num_retries=None):
        """Constructor. Instantiate once for each uploaded file.

    Args:
      tracker_callback: Callback function that takes a string argument.  Used
                        by caller to track this upload across upload
                        interruption.
      logger: logging.logger instance to use for debug messages.
      resume_url: If present, attempt to resume the upload at this URL.
      num_retries: Number of times to retry the upload making no progress.
                   This count resets every time we make progress, so the upload
                   can span many more than this number of retries.
    """
        if resume_url:
            self._SetUploadUrl(resume_url)
        else:
            self.upload_url = None
        self.num_retries = num_retries
        self.service_has_bytes = 0
        self.upload_start_point = None
        self.tracker_callback = tracker_callback
        self.logger = logger

    def _SetUploadUrl(self, url):
        """Saves URL and resets upload state.

    Called when we start a new resumable upload or get a new tracker
    URL for the upload.

    Args:
      url: URL string for the upload.

    Raises InvalidUrlError if URL is syntactically invalid.
    """
        parse_result = urllib.parse.urlparse(url)
        if parse_result.scheme.lower() not in ['http', 'https'] or not parse_result.netloc:
            raise InvalidUrlError('Invalid upload URL (%s)' % url)
        self.upload_url = url
        self.upload_url_host = config.get('Credentials', 'gs_host', None) or parse_result.netloc
        self.upload_url_path = '%s?%s' % (parse_result.path, parse_result.query)
        self.service_has_bytes = 0

    def _BuildContentRangeHeader(self, range_spec='*', length_spec='*'):
        return 'bytes %s/%s' % (range_spec, length_spec)

    def _QueryServiceState(self, conn, file_length):
        """Queries service to find out state of given upload.

    Note that this method really just makes special case use of the
    fact that the upload service always returns the current start/end
    state whenever a PUT doesn't complete.

    Args:
      conn: HTTPConnection to use for the query.
      file_length: Total length of the file.

    Returns:
      HTTP response from sending request.

    Raises:
      ResumableUploadException if problem querying service.
    """
        put_headers = {'Content-Range': self._BuildContentRangeHeader('*', file_length), 'Content-Length': '0'}
        return AWSAuthConnection.make_request(conn, 'PUT', path=self.upload_url_path, auth_path=self.upload_url_path, headers=put_headers, host=self.upload_url_host)

    def _QueryServicePos(self, conn, file_length):
        """Queries service to find out what bytes it currently has.

    Args:
      conn: HTTPConnection to use for the query.
      file_length: Total length of the file.

    Returns:
      (service_start, service_end), where the values are inclusive.
      For example, (0, 2) would mean that the service has bytes 0, 1, *and* 2.

    Raises:
      ResumableUploadException if problem querying service.
    """
        resp = self._QueryServiceState(conn, file_length)
        if resp.status == 200:
            return (0, file_length - 1)
        if resp.status != 308:
            raise ResumableUploadException('Got non-308 response (%s) from service state query' % resp.status, ResumableTransferDisposition.START_OVER)
        got_valid_response = False
        range_spec = resp.getheader('range')
        if range_spec:
            m = re.search('bytes=(\\d+)-(\\d+)', range_spec)
            if m:
                service_start = long(m.group(1))
                service_end = long(m.group(2))
                got_valid_response = True
        else:
            return self.SERVICE_HAS_NOTHING
        if not got_valid_response:
            raise ResumableUploadException("Couldn't parse upload service state query response (%s)" % str(resp.getheaders()), ResumableTransferDisposition.START_OVER)
        if conn.debug >= 1:
            self.logger.debug('Service has: Range: %d - %d.', service_start, service_end)
        return (service_start, service_end)

    def _StartNewResumableUpload(self, key, headers=None):
        """Starts a new resumable upload.

    Args:
      key: Boto Key representing the object to upload.
      headers: Headers to use in the upload requests.

    Raises:
      ResumableUploadException if any errors occur.
    """
        conn = key.bucket.connection
        if conn.debug >= 1:
            self.logger.debug('Starting new resumable upload.')
        self.service_has_bytes = 0
        post_headers = {}
        for k in headers:
            if k.lower() == 'content-length':
                raise ResumableUploadException('Attempt to specify Content-Length header (disallowed)', ResumableTransferDisposition.ABORT)
            post_headers[k] = headers[k]
        post_headers[conn.provider.resumable_upload_header] = 'start'
        resp = conn.make_request('POST', key.bucket.name, key.name, post_headers)
        body = resp.read()
        if resp.status in [429, 500, 503]:
            raise ResumableUploadException('Got status %d from attempt to start resumable upload. Will wait/retry' % resp.status, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
        elif resp.status != 200 and resp.status != 201:
            raise ResumableUploadException('Got status %d from attempt to start resumable upload. Aborting' % resp.status, ResumableTransferDisposition.ABORT)
        upload_url = resp.getheader('Location')
        if not upload_url:
            raise ResumableUploadException('No resumable upload URL found in resumable initiation POST response (%s)' % body, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
        self._SetUploadUrl(upload_url)
        self.tracker_callback(upload_url)

    def _UploadFileBytes(self, conn, http_conn, fp, file_length, total_bytes_uploaded, cb, num_cb, headers):
        """Attempts to upload file bytes.

    Makes a single attempt using an existing resumable upload connection.

    Args:
      conn: HTTPConnection from the boto Key.
      http_conn: Separate HTTPConnection for the transfer.
      fp: File pointer containing bytes to upload.
      file_length: Total length of the file.
      total_bytes_uploaded: The total number of bytes uploaded.
      cb: Progress callback function that takes (progress, total_size).
      num_cb: Granularity of the callback (maximum number of times the
              callback will be called during the file transfer). If negative,
              perform callback with each buffer read.
      headers: Headers to be used in the upload requests.

    Returns:
      (etag, generation, metageneration) from service upon success.

    Raises:
      ResumableUploadException if any problems occur.
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
        put_headers = headers.copy() if headers else {}
        if file_length:
            if total_bytes_uploaded == file_length:
                range_header = self._BuildContentRangeHeader('*', file_length)
            else:
                range_header = self._BuildContentRangeHeader('%d-%d' % (total_bytes_uploaded, file_length - 1), file_length)
            put_headers['Content-Range'] = range_header
        put_headers['Content-Length'] = str(file_length - total_bytes_uploaded)
        http_request = AWSAuthConnection.build_base_http_request(conn, 'PUT', path=self.upload_url_path, auth_path=None, headers=put_headers, host=self.upload_url_host)
        http_conn.putrequest('PUT', http_request.path)
        for k in put_headers:
            http_conn.putheader(k, put_headers[k])
        http_conn.endheaders()
        http_conn.set_debuglevel(0)
        while buf:
            if six.PY2:
                http_conn.send(buf)
                total_bytes_uploaded += len(buf)
            elif isinstance(buf, bytes):
                http_conn.send(buf)
                total_bytes_uploaded += len(buf)
            else:
                buf_bytes = buf.encode(UTF8)
                http_conn.send(buf_bytes)
                total_bytes_uploaded += len(buf_bytes)
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
        if resp.status == 200:
            return (resp.getheader('etag'), resp.getheader('x-goog-generation'), resp.getheader('x-goog-metageneration'))
        elif resp.status in [408, 429, 500, 503]:
            disposition = ResumableTransferDisposition.WAIT_BEFORE_RETRY
        else:
            disposition = ResumableTransferDisposition.ABORT
        raise ResumableUploadException('Got response code %d while attempting upload (%s)' % (resp.status, resp.reason), disposition)

    def _AttemptResumableUpload(self, key, fp, file_length, headers, cb, num_cb):
        """Attempts a resumable upload.

    Args:
      key: Boto key representing object to upload.
      fp: File pointer containing upload bytes.
      file_length: Total length of the upload.
      headers: Headers to be used in upload requests.
      cb: Progress callback function that takes (progress, total_size).
      num_cb: Granularity of the callback (maximum number of times the
              callback will be called during the file transfer). If negative,
              perform callback with each buffer read.

    Returns:
      (etag, generation, metageneration) from service upon success.

    Raises:
      ResumableUploadException if any problems occur.
    """
        service_start, service_end = self.SERVICE_HAS_NOTHING
        conn = key.bucket.connection
        if self.upload_url:
            try:
                service_start, service_end = self._QueryServicePos(conn, file_length)
                self.service_has_bytes = service_start
                if conn.debug >= 1:
                    self.logger.debug('Resuming transfer.')
            except ResumableUploadException as e:
                if conn.debug >= 1:
                    self.logger.debug('Unable to resume transfer (%s).', e.message)
                self._StartNewResumableUpload(key, headers)
        else:
            self._StartNewResumableUpload(key, headers)
        if self.upload_start_point is None:
            self.upload_start_point = service_end
        total_bytes_uploaded = service_end + 1
        if total_bytes_uploaded < file_length:
            fp.seek(total_bytes_uploaded)
        conn = key.bucket.connection
        http_conn = conn.new_http_connection(self.upload_url_host, conn.port, conn.is_secure)
        http_conn.set_debuglevel(conn.debug)
        try:
            return self._UploadFileBytes(conn, http_conn, fp, file_length, total_bytes_uploaded, cb, num_cb, headers)
        except (ResumableUploadException, socket.error):
            resp = self._QueryServiceState(conn, file_length)
            if resp.status == 400:
                raise ResumableUploadException('Got 400 response from service state query after failed resumable upload attempt. This can happen for various reasons, including specifying an invalid request (e.g., an invalid canned ACL) or if the file size changed between upload attempts', ResumableTransferDisposition.ABORT)
            else:
                raise
        finally:
            http_conn.close()

    def HandleResumableUploadException(self, e, debug):
        if e.disposition == ResumableTransferDisposition.ABORT_CUR_PROCESS:
            if debug >= 1:
                self.logger.debug('Caught non-retryable ResumableUploadException (%s); aborting but retaining tracker file', e.message)
            raise
        elif e.disposition == ResumableTransferDisposition.ABORT:
            if debug >= 1:
                self.logger.debug('Caught non-retryable ResumableUploadException (%s); aborting and removing tracker file', e.message)
            raise
        elif e.disposition == ResumableTransferDisposition.START_OVER:
            raise
        elif debug >= 1:
            self.logger.debug('Caught ResumableUploadException (%s) - will retry', e.message)

    def TrackProgressLessIterations(self, service_had_bytes_before_attempt, debug=0):
        """Tracks the number of iterations without progress.

    Performs randomized exponential backoff.

    Args:
      service_had_bytes_before_attempt: Number of bytes the service had prior
                                       to this upload attempt.
      debug: debug level 0..3
    """
        if self.service_has_bytes > service_had_bytes_before_attempt:
            self.progress_less_iterations = 0
        else:
            self.progress_less_iterations += 1
        if self.progress_less_iterations > self.num_retries:
            raise ResumableUploadException('Too many resumable upload attempts failed without progress. You might try this upload again later', ResumableTransferDisposition.ABORT_CUR_PROCESS)
        sleep_time_secs = min(random.random() * 2 ** self.progress_less_iterations, GetMaxRetryDelay())
        if debug >= 1:
            self.logger.debug('Got retryable failure (%d progress-less in a row).\nSleeping %3.1f seconds before re-trying', self.progress_less_iterations, sleep_time_secs)
        time.sleep(sleep_time_secs)

    def SendFile(self, key, fp, size, headers, canned_acl=None, cb=None, num_cb=XML_PROGRESS_CALLBACKS):
        """Upload a file to a key into a bucket on GS, resumable upload protocol.

    Args:
      key: `boto.s3.key.Key` or subclass representing the upload destination.
      fp: File pointer to upload
      size: Size of the file to upload.
      headers: The headers to pass along with the PUT request
      canned_acl: Optional canned ACL to apply to object.
      cb: Callback function that will be called to report progress on
          the upload.  The callback should accept two integer parameters, the
          first representing the number of bytes that have been successfully
          transmitted to GS, and the second representing the total number of
          bytes that need to be transmitted.
      num_cb: (optional) If a callback is specified with the cb parameter, this
              parameter determines the granularity of the callback by defining
              the maximum number of times the callback will be called during the
              file transfer. Providing a negative integer will cause your
              callback to be called with each buffer read.

    Raises:
      ResumableUploadException if a problem occurs during the transfer.
    """
        if not headers:
            headers = {}
        content_type = 'Content-Type'
        if content_type in headers and headers[content_type] is None:
            del headers[content_type]
        if canned_acl:
            headers[key.provider.acl_header] = canned_acl
        headers['User-Agent'] = UserAgent
        file_length = size
        debug = key.bucket.connection.debug
        if self.num_retries is None:
            self.num_retries = GetNumRetries()
        self.progress_less_iterations = 0
        while True:
            service_had_bytes_before_attempt = self.service_has_bytes
            try:
                _, self.generation, self.metageneration = self._AttemptResumableUpload(key, fp, file_length, headers, cb, num_cb)
                key.generation = self.generation
                if debug >= 1:
                    self.logger.debug('Resumable upload complete.')
                return
            except self.RETRYABLE_EXCEPTIONS as e:
                if debug >= 1:
                    self.logger.debug('Caught exception (%s)', e.__repr__())
                if isinstance(e, IOError) and e.errno == errno.EPIPE:
                    key.bucket.connection.connection.close()
            except ResumableUploadException as e:
                self.HandleResumableUploadException(e, debug)
            self.TrackProgressLessIterations(service_had_bytes_before_attempt, debug=debug)