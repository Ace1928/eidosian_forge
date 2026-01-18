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
class ResumableUploadHandler(object):
    BUFFER_SIZE = 8192
    RETRYABLE_EXCEPTIONS = (httplib.HTTPException, IOError, socket.error, socket.gaierror)
    SERVER_HAS_NOTHING = (0, -1)

    def __init__(self, tracker_file_name=None, num_retries=None):
        """
        Constructor. Instantiate once for each uploaded file.

        :type tracker_file_name: string
        :param tracker_file_name: optional file name to save tracker URI.
            If supplied and the current process fails the upload, it can be
            retried in a new process. If called with an existing file containing
            a valid tracker URI, we'll resume the upload from this URI; else
            we'll start a new resumable upload (and write the URI to this
            tracker file).

        :type num_retries: int
        :param num_retries: the number of times we'll re-try a resumable upload
            making no progress. (Count resets every time we get progress, so
            upload can span many more than this number of retries.)
        """
        self.tracker_file_name = tracker_file_name
        self.num_retries = num_retries
        self.server_has_bytes = 0
        self.tracker_uri = None
        if tracker_file_name:
            self._load_tracker_uri_from_file()
        self.upload_start_point = None

    def _load_tracker_uri_from_file(self):
        f = None
        try:
            f = open(self.tracker_file_name, 'r')
            uri = f.readline().strip()
            self._set_tracker_uri(uri)
        except IOError as e:
            if e.errno != errno.ENOENT:
                print("Couldn't read URI tracker file (%s): %s. Restarting upload from scratch." % (self.tracker_file_name, e.strerror))
        except InvalidUriError as e:
            print('Invalid tracker URI (%s) found in URI tracker file (%s). Restarting upload from scratch.' % (uri, self.tracker_file_name))
        finally:
            if f:
                f.close()

    def _save_tracker_uri_to_file(self):
        """
        Saves URI to tracker file if one was passed to constructor.
        """
        if not self.tracker_file_name:
            return
        f = None
        try:
            with os.fdopen(os.open(self.tracker_file_name, os.O_WRONLY | os.O_CREAT, 384), 'w') as f:
                f.write(self.tracker_uri)
        except IOError as e:
            raise ResumableUploadException("Couldn't write URI tracker file (%s): %s.\nThis can happenif you're using an incorrectly configured upload tool\n(e.g., gsutil configured to save tracker files to an unwritable directory)" % (self.tracker_file_name, e.strerror), ResumableTransferDisposition.ABORT)

    def _set_tracker_uri(self, uri):
        """
        Called when we start a new resumable upload or get a new tracker
        URI for the upload. Saves URI and resets upload state.

        Raises InvalidUriError if URI is syntactically invalid.
        """
        parse_result = urlparse.urlparse(uri)
        if parse_result.scheme.lower() not in ['http', 'https'] or not parse_result.netloc:
            raise InvalidUriError('Invalid tracker URI (%s)' % uri)
        self.tracker_uri = uri
        self.tracker_uri_host = parse_result.netloc
        self.tracker_uri_path = '%s?%s' % (parse_result.path, parse_result.query)
        self.server_has_bytes = 0

    def get_tracker_uri(self):
        """
        Returns upload tracker URI, or None if the upload has not yet started.
        """
        return self.tracker_uri

    def get_upload_id(self):
        """
        Returns the upload ID for the resumable upload, or None if the upload
        has not yet started.
        """
        delim = '?upload_id='
        if self.tracker_uri and delim in self.tracker_uri:
            return self.tracker_uri[self.tracker_uri.index(delim) + len(delim):]
        else:
            return None

    def _remove_tracker_file(self):
        if self.tracker_file_name and os.path.exists(self.tracker_file_name):
            os.unlink(self.tracker_file_name)

    def _build_content_range_header(self, range_spec='*', length_spec='*'):
        return 'bytes %s/%s' % (range_spec, length_spec)

    def _query_server_state(self, conn, file_length):
        """
        Queries server to find out state of given upload.

        Note that this method really just makes special case use of the
        fact that the upload server always returns the current start/end
        state whenever a PUT doesn't complete.

        Returns HTTP response from sending request.

        Raises ResumableUploadException if problem querying server.
        """
        put_headers = {}
        put_headers['Content-Range'] = self._build_content_range_header('*', file_length)
        put_headers['Content-Length'] = '0'
        return AWSAuthConnection.make_request(conn, 'PUT', path=self.tracker_uri_path, auth_path=self.tracker_uri_path, headers=put_headers, host=self.tracker_uri_host)

    def _query_server_pos(self, conn, file_length):
        """
        Queries server to find out what bytes it currently has.

        Returns (server_start, server_end), where the values are inclusive.
        For example, (0, 2) would mean that the server has bytes 0, 1, *and* 2.

        Raises ResumableUploadException if problem querying server.
        """
        resp = self._query_server_state(conn, file_length)
        if resp.status == 200:
            return (0, file_length - 1)
        if resp.status != 308:
            raise ResumableUploadException('Got non-308 response (%s) from server state query' % resp.status, ResumableTransferDisposition.START_OVER)
        got_valid_response = False
        range_spec = resp.getheader('range')
        if range_spec:
            m = re.search('bytes=(\\d+)-(\\d+)', range_spec)
            if m:
                server_start = long(m.group(1))
                server_end = long(m.group(2))
                got_valid_response = True
        else:
            return self.SERVER_HAS_NOTHING
        if not got_valid_response:
            raise ResumableUploadException("Couldn't parse upload server state query response (%s)" % str(resp.getheaders()), ResumableTransferDisposition.START_OVER)
        if conn.debug >= 1:
            print('Server has: Range: %d - %d.' % (server_start, server_end))
        return (server_start, server_end)

    def _start_new_resumable_upload(self, key, headers=None):
        """
        Starts a new resumable upload.

        Raises ResumableUploadException if any errors occur.
        """
        conn = key.bucket.connection
        if conn.debug >= 1:
            print('Starting new resumable upload.')
        self.server_has_bytes = 0
        post_headers = {}
        for k in headers:
            if k.lower() == 'content-length':
                raise ResumableUploadException('Attempt to specify Content-Length header (disallowed)', ResumableTransferDisposition.ABORT)
            post_headers[k] = headers[k]
        post_headers[conn.provider.resumable_upload_header] = 'start'
        resp = conn.make_request('POST', key.bucket.name, key.name, post_headers)
        body = resp.read()
        if resp.status in [500, 503]:
            raise ResumableUploadException('Got status %d from attempt to start resumable upload. Will wait/retry' % resp.status, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
        elif resp.status != 200 and resp.status != 201:
            raise ResumableUploadException('Got status %d from attempt to start resumable upload. Aborting' % resp.status, ResumableTransferDisposition.ABORT)
        tracker_uri = resp.getheader('Location')
        if not tracker_uri:
            raise ResumableUploadException('No resumable tracker URI found in resumable initiation POST response (%s)' % body, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
        self._set_tracker_uri(tracker_uri)
        self._save_tracker_uri_to_file()

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

    def _check_final_md5(self, key, etag):
        """
        Checks that etag from server agrees with md5 computed before upload.
        This is important, since the upload could have spanned a number of
        hours and multiple processes (e.g., gsutil runs), and the user could
        change some of the file and not realize they have inconsistent data.
        """
        if key.bucket.connection.debug >= 1:
            print('Checking md5 against etag.')
        if key.md5 != etag.strip('"\''):
            key.open_read()
            key.close()
            key.delete()
            raise ResumableUploadException("File changed during upload: md5 signature doesn't match etag (incorrect uploaded object deleted)", ResumableTransferDisposition.ABORT)

    def handle_resumable_upload_exception(self, e, debug):
        if e.disposition == ResumableTransferDisposition.ABORT_CUR_PROCESS:
            if debug >= 1:
                print('Caught non-retryable ResumableUploadException (%s); aborting but retaining tracker file' % e.message)
            raise
        elif e.disposition == ResumableTransferDisposition.ABORT:
            if debug >= 1:
                print('Caught non-retryable ResumableUploadException (%s); aborting and removing tracker file' % e.message)
            self._remove_tracker_file()
            raise
        elif debug >= 1:
            print('Caught ResumableUploadException (%s) - will retry' % e.message)

    def track_progress_less_iterations(self, server_had_bytes_before_attempt, roll_back_md5=True, debug=0):
        if self.server_has_bytes > server_had_bytes_before_attempt:
            self.progress_less_iterations = 0
        else:
            self.progress_less_iterations += 1
            if roll_back_md5:
                self.digesters = self.digesters_before_attempt
        if self.progress_less_iterations > self.num_retries:
            raise ResumableUploadException('Too many resumable upload attempts failed without progress. You might try this upload again later', ResumableTransferDisposition.ABORT_CUR_PROCESS)
        sleep_time_secs = random.random() * 2 ** self.progress_less_iterations
        if debug >= 1:
            print('Got retryable failure (%d progress-less in a row).\nSleeping %3.1f seconds before re-trying' % (self.progress_less_iterations, sleep_time_secs))
        time.sleep(sleep_time_secs)

    def send_file(self, key, fp, headers, cb=None, num_cb=10, hash_algs=None):
        """
        Upload a file to a key into a bucket on GS, using GS resumable upload
        protocol.

        :type key: :class:`boto.s3.key.Key` or subclass
        :param key: The Key object to which data is to be uploaded

        :type fp: file-like object
        :param fp: The file pointer to upload

        :type headers: dict
        :param headers: The headers to pass along with the PUT request

        :type cb: function
        :param cb: a callback function that will be called to report progress on
            the upload.  The callback should accept two integer parameters, the
            first representing the number of bytes that have been successfully
            transmitted to GS, and the second representing the total number of
            bytes that need to be transmitted.

        :type num_cb: int
        :param num_cb: (optional) If a callback is specified with the cb
            parameter, this parameter determines the granularity of the callback
            by defining the maximum number of times the callback will be called
            during the file transfer. Providing a negative integer will cause
            your callback to be called with each buffer read.

        :type hash_algs: dictionary
        :param hash_algs: (optional) Dictionary mapping hash algorithm
            descriptions to corresponding state-ful hashing objects that
            implement update(), digest(), and copy() (e.g. hashlib.md5()).
            Defaults to {'md5': md5()}.

        Raises ResumableUploadException if a problem occurs during the transfer.
        """
        if not headers:
            headers = {}
        CT = 'Content-Type'
        if CT in headers and headers[CT] is None:
            del headers[CT]
        headers['User-Agent'] = UserAgent
        if isinstance(fp, KeyFile):
            file_length = fp.getkey().size
        else:
            fp.seek(0, os.SEEK_END)
            file_length = fp.tell()
            fp.seek(0)
        debug = key.bucket.connection.debug
        if hash_algs is None:
            hash_algs = {'md5': md5}
        self.digesters = dict(((alg, hash_algs[alg]()) for alg in hash_algs or {}))
        if self.num_retries is None:
            self.num_retries = config.getint('Boto', 'num_retries', 6)
        self.progress_less_iterations = 0
        while True:
            server_had_bytes_before_attempt = self.server_has_bytes
            self.digesters_before_attempt = dict(((alg, self.digesters[alg].copy()) for alg in self.digesters))
            try:
                etag, self.generation, self.metageneration = self._attempt_resumable_upload(key, fp, file_length, headers, cb, num_cb)
                for alg in self.digesters:
                    key.local_hashes[alg] = self.digesters[alg].digest()
                self._remove_tracker_file()
                self._check_final_md5(key, etag)
                key.generation = self.generation
                if debug >= 1:
                    print('Resumable upload complete.')
                return
            except self.RETRYABLE_EXCEPTIONS as e:
                if debug >= 1:
                    print('Caught exception (%s)' % e.__repr__())
                if isinstance(e, IOError) and e.errno == errno.EPIPE:
                    key.bucket.connection.connection.close()
            except ResumableUploadException as e:
                self.handle_resumable_upload_exception(e, debug)
            self.track_progress_less_iterations(server_had_bytes_before_attempt, True, debug)