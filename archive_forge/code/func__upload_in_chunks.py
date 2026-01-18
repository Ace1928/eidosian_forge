import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _upload_in_chunks(self, stream, object_path, lease, meta_data, content_type, object_name, file_path, verify_hash, headers):
    """
        Uploads data from an iterator in fixed sized chunks to Azure Storage
        """
    data_hash = None
    if verify_hash:
        data_hash = self._get_hash_function()
    bytes_transferred = 0
    count = 1
    chunks = []
    headers = headers or {}
    lease.update_headers(headers)
    params = {'comp': 'block'}
    for data in read_in_chunks(stream, AZURE_UPLOAD_CHUNK_SIZE, fill_size=True):
        data = b(data)
        content_length = len(data)
        bytes_transferred += content_length
        if verify_hash:
            data_hash.update(data)
        chunk_hash = self._get_hash_function()
        chunk_hash.update(data)
        chunk_hash = base64.b64encode(b(chunk_hash.digest()))
        headers['Content-MD5'] = chunk_hash.decode('utf-8')
        headers['Content-Length'] = str(content_length)
        block_id = base64.b64encode(b('%10d' % count))
        block_id = block_id.decode('utf-8')
        params['blockid'] = block_id
        chunks.append(block_id)
        lease.renew()
        resp = self.connection.request(object_path, method='PUT', data=data, headers=headers, params=params)
        if resp.status != httplib.CREATED:
            resp.parse_error()
            raise LibcloudError('Error uploading chunk %d. Code: %d' % (count, resp.status), driver=self)
        count += 1
    if verify_hash:
        data_hash = base64.b64encode(b(data_hash.digest()))
        data_hash = data_hash.decode('utf-8')
    response = self._commit_blocks(object_path=object_path, chunks=chunks, lease=lease, headers=headers, meta_data=meta_data, content_type=content_type, data_hash=data_hash, object_name=object_name, file_path=file_path)
    response.headers['content-md5'] = None
    return {'response': response, 'data_hash': data_hash, 'bytes_transferred': bytes_transferred}