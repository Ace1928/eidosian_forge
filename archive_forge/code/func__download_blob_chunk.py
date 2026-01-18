import base64
import io
import logging
import smart_open.bytebuffer
import smart_open.constants
def _download_blob_chunk(self, size):
    if self._size == self._position:
        return b''
    elif size == -1:
        stream = self._blob.download_blob(offset=self._position, max_concurrency=self._concurrency)
    else:
        stream = self._blob.download_blob(offset=self._position, max_concurrency=self._concurrency, length=size)
    logging.debug('reading with a max concurrency of %d', self._concurrency)
    if isinstance(stream, azure.storage.blob.StorageStreamDownloader):
        binary = stream.readall()
    else:
        binary = stream.read()
    return binary