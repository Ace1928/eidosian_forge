import io
import logging
import urllib.parse
from smart_open import utils, constants
import http.client as httplib
class BufferedOutputBase(io.BufferedIOBase):

    def __init__(self, uri, min_part_size=MIN_PART_SIZE):
        """
        Parameters
        ----------
        min_part_size: int, optional
            For writing only.

        """
        self._uri = uri
        self._closed = False
        self.min_part_size = min_part_size
        payload = {'op': 'CREATE', 'overwrite': True}
        init_response = requests.put(self._uri, params=payload, allow_redirects=False)
        if not init_response.status_code == httplib.TEMPORARY_REDIRECT:
            raise WebHdfsException.from_response(init_response)
        uri = init_response.headers['location']
        response = requests.put(uri, data='', headers={'content-type': 'application/octet-stream'})
        if not response.status_code == httplib.CREATED:
            raise WebHdfsException.from_response(response)
        self.lines = []
        self.parts = 0
        self.chunk_bytes = 0
        self.total_size = 0
        self.raw = None

    def writable(self):
        """Return True if the stream supports writing."""
        return True

    def detach(self):
        raise io.UnsupportedOperation('detach() not supported')

    def _upload(self, data):
        payload = {'op': 'APPEND'}
        init_response = requests.post(self._uri, params=payload, allow_redirects=False)
        if not init_response.status_code == httplib.TEMPORARY_REDIRECT:
            raise WebHdfsException.from_response(init_response)
        uri = init_response.headers['location']
        response = requests.post(uri, data=data, headers={'content-type': 'application/octet-stream'})
        if not response.status_code == httplib.OK:
            raise WebHdfsException.from_response(response)

    def write(self, b):
        """
        Write the given bytes (binary string) into the WebHDFS file from constructor.

        """
        if self._closed:
            raise ValueError('I/O operation on closed file')
        if not isinstance(b, bytes):
            raise TypeError('input must be a binary string')
        self.lines.append(b)
        self.chunk_bytes += len(b)
        self.total_size += len(b)
        if self.chunk_bytes >= self.min_part_size:
            buff = b''.join(self.lines)
            logger.info('uploading part #%i, %i bytes (total %.3fGB)', self.parts, len(buff), self.total_size / 1024.0 ** 3)
            self._upload(buff)
            logger.debug('upload of part #%i finished', self.parts)
            self.parts += 1
            self.lines, self.chunk_bytes = ([], 0)

    def close(self):
        buff = b''.join(self.lines)
        if buff:
            logger.info('uploading last part #%i, %i bytes (total %.3fGB)', self.parts, len(buff), self.total_size / 1024.0 ** 3)
            self._upload(buff)
            logger.debug('upload of last part #%i finished', self.parts)
        self._closed = True

    @property
    def closed(self):
        return self._closed