import os
import hmac
import time
import base64
import codecs
from hashlib import sha1
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _upload_from_iterator(self, iterator, object_path, upload_id, calculate_hash=True, container=None):
    """
        Uploads data from an iterator in fixed sized chunks to OSS

        :param iterator: The generator for fetching the upload data
        :type iterator: ``generator``

        :param object_path: The path of the object to which we are uploading
        :type object_name: ``str``

        :param upload_id: The upload id allocated for this multipart upload
        :type upload_id: ``str``

        :keyword calculate_hash: Indicates if we must calculate the data hash
        :type calculate_hash: ``bool``

        :keyword container: the container object to upload object to
        :type container: :class:`Container`

        :return: A tuple of (chunk info, checksum, bytes transferred)
        :rtype: ``tuple``
        """
    data_hash = None
    if calculate_hash:
        data_hash = self._get_hash_function()
    bytes_transferred = 0
    count = 1
    chunks = []
    params = {'uploadId': upload_id}
    for data in read_in_chunks(iterator, chunk_size=CHUNK_SIZE, fill_size=True, yield_empty=True):
        bytes_transferred += len(data)
        if calculate_hash:
            data_hash.update(data)
        chunk_hash = self._get_hash_function()
        chunk_hash.update(data)
        chunk_hash = base64.b64encode(chunk_hash.digest()).decode('utf-8')
        headers = {'Content-MD5': chunk_hash}
        params['partNumber'] = count
        request_path = '?'.join((object_path, urlencode(params)))
        resp = self.connection.request(request_path, method='PUT', data=data, headers=headers, container=container)
        if resp.status != httplib.OK:
            raise LibcloudError('Error uploading chunk', driver=self)
        server_hash = resp.headers['etag']
        chunks.append((count, server_hash))
        count += 1
    if calculate_hash:
        data_hash = data_hash.hexdigest()
    return (chunks, data_hash, bytes_transferred)